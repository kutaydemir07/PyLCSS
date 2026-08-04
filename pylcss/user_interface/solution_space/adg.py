# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""AllDimensionsGraphMixin behavior for solution-space analysis."""

from __future__ import annotations

import logging

import networkx as nx
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


from .plotting import (
    ArrowLine,
    ScalableText,
)

logger = logging.getLogger(__name__)

__all__ = ["AllDimensionsGraphMixin"]


class AdgLayoutWorker(QtCore.QThread):
    """Calculate a potentially expensive NetworkX layout off the GUI thread."""

    result_signal = QtCore.Signal(object)
    error_signal = QtCore.Signal(str)

    def __init__(self, graph, scope_label):
        super().__init__()
        self.graph = graph
        self.scope_label = scope_label

    def run(self):
        try:
            positions = AllDimensionsGraphMixin._hierarchical_layout(self.graph)
        except Exception as exc:
            logger.exception("ADG layout failed")
            self.error_signal.emit(str(exc))
        else:
            self.result_signal.emit(positions)


class AllDimensionsGraphMixin:
    def compute_adg(self):
        """Compute and visualize the Attribute Dependency Graph from the system model."""
        if not self.problem:
            QtWidgets.QMessageBox.warning(self, "Warning", "No system loaded.")
            return

        # Determine scope selection if available
        selected_scope = None
        if hasattr(self, "combo_adg_scope") and isinstance(
            self.combo_adg_scope, QtWidgets.QComboBox
        ):
            selected_scope = self.combo_adg_scope.currentText()

        # Get the graph from the main application's modeling widget
        try:
            main_window = self.window()
            if not hasattr(main_window, "modeling_widget"):
                # Try to refresh list just in case, but usually this means we are standalone
                if hasattr(self, "refresh_adg_system_list"):
                    self.refresh_adg_system_list()
                if selected_scope is None or (
                    hasattr(self, "combo_adg_scope")
                    and self.combo_adg_scope.count() <= 1
                ):
                    QtWidgets.QMessageBox.warning(
                        self, "Warning", "Modeling environment not available."
                    )
                    return

            modeling_widget = main_window.modeling_widget
            if not hasattr(modeling_widget, "system_manager"):
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "System manager not found."
                )
                return

            # Get systems
            systems = modeling_widget.system_manager.systems
            if not systems:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Warning",
                    "No systems in Modeling Environment. Create a system first.",
                )
                return

            # Create NetworkX graph (single or merged depending on scope)
            G = nx.DiGraph()
            node_to_name = {}

            # Select target systems based on scope
            target_systems = []
            if selected_scope in (None, "Merged System"):
                target_systems = systems
            else:
                for sys in systems:
                    # Systems are dictionaries with 'name' key
                    sys_name = (
                        sys.get("name")
                        if isinstance(sys, dict)
                        else getattr(sys, "name", None)
                    )
                    if sys_name == selected_scope:
                        target_systems = [sys]
                        break

            if not target_systems:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", f"System '{selected_scope}' not found."
                )
                return

            # Process systems
            for sys_idx, system in enumerate(target_systems):
                graph = system.get("graph")
                if not graph:
                    continue

                # Get all nodes from this system's graph
                nodes = graph.all_nodes()

                # Categorize nodes
                input_nodes = [n for n in nodes if n.type_.startswith("com.pfd.input")]
                output_nodes = [
                    n for n in nodes if n.type_.startswith("com.pfd.output")
                ]
                intermediate_nodes = [
                    n for n in nodes if n.type_.startswith("com.pfd.intermediate")
                ]
                blackbox_nodes = [
                    n for n in nodes if n.type_.startswith("com.pfd.custom_block")
                ]

                # Add input nodes (design variables)
                for n in input_nodes:
                    if n.has_property("input_props"):
                        var_name = n.get_property("input_props").get(
                            "var_name", n.name()
                        )
                    else:
                        var_name = n.get_property("var_name") or n.name()
                    node_to_name[n.id] = var_name
                    if not G.has_node(var_name):
                        G.add_node(var_name, node_type="input")

                # Add output nodes (QoIs)
                for n in output_nodes:
                    if n.has_property("output_props"):
                        var_name = n.get_property("output_props").get(
                            "var_name", n.name()
                        )
                    else:
                        var_name = n.get_property("var_name") or n.name()
                    node_to_name[n.id] = var_name
                    if not G.has_node(var_name):
                        G.add_node(var_name, node_type="output")

                # Add intermediate variable nodes
                for n in intermediate_nodes:
                    var_name = n.get_property("var_name") or n.name()
                    # Make unique across systems if merging
                    unique_name = var_name
                    if selected_scope in (None, "Merged System"):
                        counter = 1
                        while G.has_node(unique_name):
                            unique_name = f"{var_name}_{counter}"
                            counter += 1
                    node_to_name[n.id] = unique_name
                    G.add_node(unique_name, node_type="intermediate")

                # Add blackbox function nodes (small black dots)
                for n in blackbox_nodes:
                    func_name = n.name()
                    # Make unique across systems if merging
                    unique_name = func_name
                    if selected_scope in (None, "Merged System"):
                        counter = 1
                        while G.has_node(unique_name):
                            unique_name = f"{func_name}_{counter}"
                            counter += 1
                    node_to_name[n.id] = unique_name
                    G.add_node(unique_name, node_type="blackbox")

                # Build edges from connections in this system's graph
                for n in nodes:
                    source_name = node_to_name.get(n.id)
                    if source_name:
                        for port in n.output_ports():
                            for connected_port in port.connected_ports():
                                target_node = connected_port.node()
                                target_name = node_to_name.get(target_node.id)
                                if target_name:
                                    G.add_edge(source_name, target_name)

            if G.number_of_nodes() == 0:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "No nodes found in system graphs."
                )
                return

            self._start_adg_layout(G, selected_scope or "Merged System")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to compute ADG: {str(e)}"
            )
            logger.exception("ADG computation failed")

    def refresh_adg_system_list(self):
        """Populate the ADG scope combo box with available systems."""
        if not hasattr(self, "combo_adg_scope"):
            return
        current_text = (
            self.combo_adg_scope.currentText()
            if self.combo_adg_scope.count() > 0
            else "Merged System"
        )
        self.combo_adg_scope.blockSignals(True)
        self.combo_adg_scope.clear()
        self.combo_adg_scope.addItem("Merged System")

        # Try to get systems from modeling widget if available
        try:
            main_window = self.window()
            if hasattr(main_window, "modeling_widget") and hasattr(
                main_window.modeling_widget, "system_manager"
            ):
                systems = main_window.modeling_widget.system_manager.systems
                for sys in systems:
                    # Systems are dictionaries with 'name' key
                    name = (
                        sys.get("name")
                        if isinstance(sys, dict)
                        else getattr(sys, "name", None)
                    )
                    if name:
                        self.combo_adg_scope.addItem(name)
        except Exception as e:
            logger.warning(f"Failed to refresh ADG system list: {e}")

        # Restore selection if possible
        idx = self.combo_adg_scope.findText(current_text)
        if idx >= 0:
            self.combo_adg_scope.setCurrentIndex(idx)
        self.combo_adg_scope.blockSignals(False)

    def _start_adg_layout(self, graph, scope_label):
        worker = getattr(self, "adg_layout_worker", None)
        if worker is not None and worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "ADG", "A graph layout is already being generated."
            )
            return
        worker = AdgLayoutWorker(graph, scope_label)
        self.adg_layout_worker = worker
        self.btn_compute_adg.setEnabled(False)
        self.lbl_adg_info.setText("Calculating graph layout...")
        worker.result_signal.connect(
            lambda positions, active_worker=worker: self._on_adg_layout_ready(
                active_worker, positions
            )
        )
        worker.error_signal.connect(
            lambda message, active_worker=worker: self._on_adg_layout_error(
                active_worker, message
            )
        )
        worker.finished.connect(
            lambda active_worker=worker: self._on_adg_layout_stopped(active_worker)
        )
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_adg_layout_ready(self, worker, positions):
        if worker is not getattr(self, "adg_layout_worker", None):
            return
        graph = worker.graph
        self.visualize_adg(graph, positions)
        for index in range(self.right_tabs.count()):
            if self.right_tabs.tabText(index) == "ADG":
                self.right_tabs.setCurrentIndex(index)
                break
        counts = {
            node_type: sum(
                graph.nodes[node].get("node_type") == node_type
                for node in graph.nodes()
            )
            for node_type in ("input", "output", "intermediate", "blackbox")
        }
        self.lbl_adg_info.setText(
            f"Scope: {worker.scope_label} | {graph.number_of_nodes()} nodes "
            f"({counts['input']} inputs, {counts['output']} outputs, "
            f"{counts['intermediate']} intermediate, {counts['blackbox']} functions), "
            f"{graph.number_of_edges()} connections"
        )

    def _on_adg_layout_error(self, worker, message):
        if worker is getattr(self, "adg_layout_worker", None):
            QtWidgets.QMessageBox.warning(
                self, "Warning", f"Graph layout failed: {message}"
            )

    def _on_adg_layout_stopped(self, worker):
        if worker is getattr(self, "adg_layout_worker", None):
            self.adg_layout_worker = None
            self.btn_compute_adg.setEnabled(True)

    def visualize_adg(self, G, pos=None):
        """Visualize the attribute dependency graph using pyqtgraph."""
        self.adg_plot.clear()

        if G.number_of_nodes() == 0:
            return

        try:
            # Calculate hierarchical layout
            if pos is None:
                pos = self._hierarchical_layout(G)

            # --- All sizes in Data Coordinates (scale with zoom) ---
            SIZE_STD = 0.8  # Standard node diameter
            SIZE_BB = 0.25  # Blackbox node diameter
            TEXT_SIZE = 0.25  # Text height in data coords
            LINE_WIDTH = 0.02  # Line width in data coords
            ARROW_HEAD = 0.12  # Arrow head length in data coords

            # Radii for arrow calculation
            RADIUS_STD = SIZE_STD / 2
            RADIUS_BB = SIZE_BB / 2

            # Draw edges with arrows (drawn first, so nodes appear on top)
            for edge in G.edges(data=True):
                source = edge[0]
                target = edge[1]
                x1, y1 = pos[source]
                x2, y2 = pos[target]

                # Determine node radius based on target type
                target_type = G.nodes[target].get("node_type", "unknown")
                target_radius = RADIUS_BB if target_type == "blackbox" else RADIUS_STD

                # Create arrow line with data-coordinate sizing
                edge_item = ArrowLine(
                    start_pos=(x1, y1),
                    end_pos=(x2, y2),
                    pen="k",
                    head_len=ARROW_HEAD,
                    node_radius=target_radius,
                    line_width=LINE_WIDTH,
                )
                self.adg_plot.addItem(edge_item)

            # Draw nodes (drawn after edges so they appear on top)
            for node in G.nodes(data=True):
                name = node[0]
                x, y = pos[name]
                node_type = node[1].get("node_type", "unknown")

                if node_type == "blackbox":
                    # Draw small black dot for blackbox functions
                    node_scatter = pg.ScatterPlotItem(
                        [x],
                        [y],
                        size=SIZE_BB,
                        pen=pg.mkPen("k", width=1),
                        brush=pg.mkBrush("k"),
                        symbol="o",
                        pxMode=False,  # Scales with zoom
                    )
                    self.adg_plot.addItem(node_scatter)
                else:
                    # Draw white circle with black border
                    node_scatter = pg.ScatterPlotItem(
                        [x],
                        [y],
                        size=SIZE_STD,
                        pen=pg.mkPen("k", width=2),
                        brush=pg.mkBrush("w"),
                        symbol="o",
                        pxMode=False,  # Scales with zoom
                    )
                    self.adg_plot.addItem(node_scatter)

                    # Add scalable text label (truncate long names)
                    display_name = name if len(name) <= 8 else name[:7] + "…"
                    # Adjust text size based on name length
                    text_size = TEXT_SIZE if len(display_name) <= 4 else TEXT_SIZE * 0.8
                    text_item = ScalableText(
                        display_name, x, y, size=text_size, color="k"
                    )
                    self.adg_plot.addItem(text_item)

            # Auto-fit view to show all nodes with padding
            if pos:
                all_x = [p[0] for p in pos.values()]
                all_y = [p[1] for p in pos.values()]
                x_min, x_max = min(all_x) - 1, max(all_x) + 1
                y_min, y_max = min(all_y) - 0.5, max(all_y) + 0.5
                self.adg_plot.setXRange(x_min, x_max, padding=0.1)
                self.adg_plot.setYRange(y_min, y_max, padding=0.1)

            # Store for later reference
            self.adg_graph = G
            self.adg_positions = pos

        except Exception as e:
            logger.exception("Graph visualization failed")
            QtWidgets.QMessageBox.warning(
                self, "Warning", f"Graph layout failed: {str(e)}"
            )

    @staticmethod
    def _hierarchical_layout(G):
        """Create a hierarchical layout optimized for ADG visualization."""
        pos = {}

        # Base spacing parameters (relative to node diameter of 0.8)
        NODE_DIAMETER = 0.8
        H_SPACING = NODE_DIAMETER * 1.5  # Horizontal spacing between nodes

        # If graph is directed, use topological levels
        if G.is_directed():
            try:
                # Compute levels: inputs (DVs) at bottom, outputs (QoIs) at top
                levels = {}

                # First pass: assign levels based on topological order
                for node in nx.topological_sort(G):
                    predecessors = list(G.predecessors(node))
                    if not predecessors:
                        levels[node] = 0
                    else:
                        levels[node] = max(levels[p] for p in predecessors) + 1

                # Second pass: ensure all sink nodes are at max level
                max_level = max(levels.values()) if levels else 0
                for node in G.nodes():
                    if G.out_degree(node) == 0:
                        levels[node] = max_level

                # Organize nodes by level
                nodes_by_level = {}
                for node, level in levels.items():
                    if level not in nodes_by_level:
                        nodes_by_level[level] = []
                    nodes_by_level[level].append(node)

                # Calculate widest level to determine aspect ratio
                max_nodes_in_level = (
                    max(len(nodes) for nodes in nodes_by_level.values())
                    if nodes_by_level
                    else 1
                )
                num_levels = len(nodes_by_level)

                # Total horizontal extent
                total_width = (
                    (max_nodes_in_level - 1) * H_SPACING
                    if max_nodes_in_level > 1
                    else H_SPACING
                )

                # Make vertical spacing proportional to create good aspect ratio
                # Target: roughly square-ish aspect ratio (height ~ width * 0.7-1.0)
                if num_levels > 1:
                    V_SPACING = max(
                        NODE_DIAMETER * 2.0, total_width * 0.6 / (num_levels - 1)
                    )
                else:
                    V_SPACING = NODE_DIAMETER * 2.0

                max(nodes_by_level.keys()) if nodes_by_level else 0

                pos = {}
                for level, nodes in nodes_by_level.items():
                    # Sort nodes to minimize crossings with previous level
                    if level > 0:
                        node_scores = {}
                        for node in nodes:
                            preds = list(G.predecessors(node))
                            if preds and all(p in pos for p in preds):
                                avg_x = sum(pos[p][0] for p in preds) / len(preds)
                                node_scores[node] = avg_x
                            else:
                                node_scores[node] = 0
                        nodes = sorted(nodes, key=lambda n: node_scores.get(n, 0))

                    # Vertical position
                    y = level * V_SPACING

                    # Horizontal spacing
                    width = len(nodes)
                    for i, node in enumerate(nodes):
                        x = (i - (width - 1) / 2.0) * H_SPACING
                        pos[node] = (x, y)

            except nx.NetworkXException:
                # Fallback if topological sort fails (graph has cycles)
                pos = nx.spring_layout(G, k=2.5, iterations=50)
        else:
            # For undirected graphs, use spring layout with increased spacing
            pos = nx.spring_layout(G, k=2.5, iterations=50)

        return pos

    def save_adg_graph(self):
        """Save the ADG graph visualization."""
        if not hasattr(self, "adg_graph"):
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No graph to save. Generate graph first."
            )
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Graph",
            "",
            "PNG Image (*.png);;GraphML (*.graphml);;All Files (*)",
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".graphml"):
                # Save graph structure
                nx.write_graphml(self.adg_graph, file_path)
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Graph saved to {file_path}"
                )
            else:
                # Save plot image using QPixmap
                # Get the scene from the plot widget
                scene = self.adg_plot.sceneObj
                # Create a QImage to render into
                scene_rect = scene.itemsBoundingRect()
                image = QtGui.QImage(
                    max(1, int(scene_rect.width())),
                    max(1, int(scene_rect.height())),
                    QtGui.QImage.Format_ARGB32,
                )
                image.fill(QtCore.Qt.white)

                # Render the scene to the image
                painter = QtGui.QPainter(image)
                scene.render(painter, QtCore.QRectF(image.rect()), scene_rect)
                painter.end()

                # Save the image
                if not image.save(file_path):
                    raise RuntimeError("Qt could not write the graph image.")
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Graph image saved to {file_path}"
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to save graph: {str(e)}"
            )
