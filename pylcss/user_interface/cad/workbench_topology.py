# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""WorkbenchTopologyMixin behavior for the Design Studio workbench."""

from __future__ import annotations

import logging
import time

import numpy as np
from PySide6 import QtCore, QtWidgets


from .execution_workers import GraphExecutionWorker

logger = logging.getLogger(__name__)

__all__ = ["WorkbenchTopologyMixin"]


class WorkbenchTopologyMixin:
    @staticmethod
    def _port_has_connections(node, port_name):
        try:
            port = node.get_input(port_name)
            return bool(port and port.connected_ports())
        except Exception:
            return False

    def _topopt_preflight_error(self, node):
        if getattr(node, "__identifier__", "") != "com.cad.sim.topopt_voxel":
            return None
        # The CAD design domain is mandatory. The topology node voxelizes it
        # internally, so a separate finite-element mesh node is not part of
        # the study definition.
        if not self._port_has_connections(node, "design_domain"):
            return (
                "Topology Opt needs a design domain. Connect a CAD solid or "
                "watertight imported surface directly to 'design_domain'; the "
                "voxel analysis grid is generated internally."
            )
        if not self._port_has_connections(node, "material"):
            return (
                "Topology Opt needs a Material connection so stiffness, stress, "
                "mass, and downstream FEA use consistent units."
            )

        # Physics-specific study nodes are required explicitly; no hidden
        # default support, load, or heat condition is fabricated.
        goal = str(node.get_property("design_goal") or "").strip().lower()
        physics = str(node.get_property("physics_mode") or "Structural").strip().lower()
        if goal == "thermal conduction":
            physics = "thermal"
        elif goal == "thermo-mechanical":
            physics = "thermo-mechanical"
        elif goal in {
            "lightweight stiffness",
            "minimum mass under stress",
            "multibody load envelope",
        }:
            physics = "structural"

        if physics in {"structural", "thermo-mechanical"}:
            if goal == "multibody load envelope":
                if not self._port_has_connections(node, "load_cases"):
                    return (
                        "Multi-body TopOpt needs connected TopOpt Operating "
                        "Case nodes containing their supports and loads."
                    )
                if not self._port_has_connections(node, "joints"):
                    return (
                        "Multi-body TopOpt needs at least one connected TopOpt "
                        "Joint with both anchors selected on the design domain."
                    )
            else:
                if not self._port_has_connections(node, "supports"):
                    return (
                        "Structural TopOpt needs a connected TopOpt Support "
                        "placed on a selected design-domain face."
                    )
                if not self._port_has_connections(node, "loads"):
                    return (
                        "Structural TopOpt needs a connected TopOpt Force "
                        "placed on a selected design-domain face."
                    )

        if physics in {"thermal", "thermo-mechanical"}:
            if not self._port_has_connections(node, "thermal_sinks"):
                return (
                    "Thermal TopOpt needs a connected TopOpt Thermal Sink "
                    "placed on a selected design-domain face."
                )
            if not self._port_has_connections(node, "thermal_loads"):
                return (
                    "Thermal TopOpt needs a connected TopOpt Heat Load "
                    "placed on a selected design-domain face."
                )

        return None

    @staticmethod
    def _topopt_cached_domain_value(node):
        try:
            port = node.get_input("design_domain")
            connected = list(port.connected_ports()) if port else []
        except Exception:
            connected = []
        if not connected:
            return None
        source_port = connected[0]
        source = source_port.node()
        value = getattr(source, "_last_result", None)
        if isinstance(value, dict):
            try:
                output_name = source_port.name()
            except Exception:
                output_name = ""
            if output_name and output_name in value:
                value = value[output_name]
            elif "shape" in value:
                value = value["shape"]
        return value

    @staticmethod
    def _topopt_spans_from_value(value):
        try:
            import numpy as np

            if isinstance(value, dict):
                if "mesh" in value:
                    return WorkbenchTopologyMixin._topopt_spans_from_value(
                        value["mesh"]
                    )
                if "vertices" in value:
                    pts = np.asarray(value["vertices"], dtype=float)
                    if pts.ndim == 2 and pts.shape[1] >= 3 and len(pts) > 0:
                        spans = pts[:, :3].max(axis=0) - pts[:, :3].min(axis=0)
                        positive = spans[spans > 1e-9]
                        if positive.size:
                            return np.where(spans > 1e-9, spans, float(positive.min()))
            if hasattr(value, "p"):
                pts = np.asarray(value.p, dtype=float)
                if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] > 0:
                    spans = pts[:3].max(axis=1) - pts[:3].min(axis=1)
                    positive = spans[spans > 1e-9]
                    if positive.size:
                        return np.where(spans > 1e-9, spans, float(positive.min()))
        except Exception:
            return None
        return None

    @staticmethod
    def _topopt_industrial_grid_from_spans(spans):
        spans = np.asarray(spans, dtype=float)
        if spans.shape[0] < 3 or not np.all(np.isfinite(spans)):
            spans = np.asarray([80.0, 28.0, 8.0])
        positive = spans[spans > 1e-9]
        if positive.size == 0:
            spans = np.asarray([80.0, 28.0, 8.0])

        target_cells = 24000.0
        max_cells = 50000
        min_axis = 6
        max_axis = 160

        voxel = max(float(np.prod(spans[:3]) / target_cells) ** (1.0 / 3.0), 1e-9)
        dims = np.ceil(spans[:3] / voxel).astype(int)
        dims = np.maximum(dims, min_axis)
        if int(dims.max()) > max_axis:
            dims = np.maximum(
                np.floor(dims * (max_axis / float(dims.max()))).astype(int),
                min_axis,
            )
        while int(np.prod(dims)) > max_cells and int(dims.max()) > min_axis:
            scale = (max_cells / float(np.prod(dims))) ** (1.0 / 3.0) * 0.98
            dims = np.maximum(np.floor(dims * scale).astype(int), min_axis)
        return [int(v) for v in dims]

    def _apply_topopt_industrial_defaults(self, node):
        if getattr(node, "__identifier__", "") != "com.cad.sim.topopt_voxel":
            return

        spans = self._topopt_spans_from_value(self._topopt_cached_domain_value(node))
        if spans is None:
            try:
                spans = [
                    max(1, int(node.get_property("nelx") or 80)),
                    max(1, int(node.get_property("nely") or 28)),
                    max(1, int(node.get_property("nelz") or 8)),
                ]
            except Exception:
                spans = [80, 28, 8]
        nelx, nely, nelz = self._topopt_industrial_grid_from_spans(spans)
        stress_enabled = str(node.get_property("stress_constraint") or "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        goal = str(node.get_property("design_goal") or "").lower()
        stress_goal = "stress" in goal
        physics_mode = (
            "Thermal"
            if goal == "thermal conduction"
            else "Thermo-Mechanical"
            if goal == "thermo-mechanical"
            else "Structural"
        )
        settings = {
            "advanced_settings_visible": False,
            "formulation": "Density (SIMP)",
            "nelx": nelx,
            "nely": nely,
            "nelz": nelz,
            "rmin": round(max(1.2, min(5.0, max(nelx, nely, nelz) * 0.030)), 2),
            "penal": 3.0,
            "density_cutoff": 0.45,
            "optimizer": (
                "GCMMA"
                if stress_enabled or stress_goal
                else "MMA"
                if goal in {"thermo-mechanical", "multibody load envelope"}
                else "Auto"
            ),
            "physics_mode": physics_mode,
            "load_aggregation": (
                "Worst Case" if goal == "multibody load envelope" else "Weighted Sum"
            ),
            "max_iter": 100,
            "tol": 0.005,
            "convergence_patience": 5,
            "print_ready_mesh": False,
            "mesh_decimate_ratio": 1.0,
            "surface_recovery_method": "Volume-Preserving SDF (VTK)",
            "structure_mode": "Solid Envelope",
            "structure_cell_size_voxels": 8.0,
            "structure_member_thickness_voxels": 1.0,
            "structure_skin_thickness_voxels": 0.75,
        }
        for key, value in settings.items():
            try:
                node.set_property(key, value)
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)

    def _upstream_closure(self, node):
        """Return *node* plus every node transitively feeding its inputs.

        This is the dependency subgraph needed to compute *node* — its
        upstream ancestors only, never downstream consumers or sibling
        branches.  Used to scope a "Run" to the selected node so that, e.g.,
        running a Mesh node does not also trigger a downstream Topology
        Optimization, and running an FEA Solver does not run a sibling TopOpt
        that merely shares the same geometry.
        """
        seen = set()
        order = []
        stack = [node]
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            order.append(n)
            try:
                ports = n.input_ports()
                if isinstance(ports, dict):
                    ports = list(ports.values())
            except Exception:
                ports = []
            for port in ports:
                try:
                    conns = list(port.connected_ports())
                except Exception:
                    conns = []
                for cp in conns:
                    try:
                        up = cp.node()
                    except Exception:
                        continue
                    if id(up) not in seen:
                        stack.append(up)
        return order

    def _run_action(self):
        """Toolbar "Run".

        If exactly one node is selected, run only that node and its upstream
        dependency chain (so siblings/downstream — e.g. a Topology Opt that
        shares the geometry — are NOT executed).  With no single selection,
        run the whole graph as before.
        """
        try:
            selected = list(self.graph.selected_nodes())
        except Exception:
            selected = []
        terminal_ids = {
            "com.cad.sim.solver",
            "com.cad.sim.crash_solver",
            "com.cad.sim.radioss_deck",
            "com.cad.sim.topopt_voxel",
        }
        selected_terminals = [
            node
            for node in selected
            if getattr(node, "__identifier__", "") in terminal_ids
        ]

        target = None
        if len(selected) == 1:
            target = selected[0]
        elif len(selected_terminals) == 1:
            target = selected_terminals[0]
        elif not selected:
            all_terminals = [
                node
                for node in self.graph.all_nodes()
                if getattr(node, "__identifier__", "") in terminal_ids
            ]
            if len(all_terminals) == 1:
                target = all_terminals[0]
            elif len(all_terminals) > 1:
                message = (
                    "This graph contains multiple independent studies. Select "
                    "the FEA, crash, or topology solver you want to run."
                )
                self.statusBar().showMessage(message)
                QtWidgets.QMessageBox.information(self, "Choose a Workflow", message)
                return
        elif len(selected_terminals) > 1:
            message = (
                "Select one terminal solver at a time so only that workflow is run."
            )
            self.statusBar().showMessage(message)
            QtWidgets.QMessageBox.information(self, "Choose a Workflow", message)
            return

        if target is not None:
            scoped = self._upstream_closure(target)
            self._last_rendered_node = target
            self._last_rendered_geom_id = None
            try:
                name = target.name() if callable(target.name) else target.name
            except Exception:
                name = "node"
            self.statusBar().showMessage(
                f"Running '{name}' and its inputs ({len(scoped)} nodes)..."
            )
            self._execute_graph(nodes=scoped)
        else:
            self._execute_graph()

    def _execute_graph(self, skip_simulation=False, nodes=None):
        """Start graph execution in a background thread.

        Args:
            skip_simulation: If True, skip FEA/TopOpt nodes (for auto-update mode)
            nodes: Optional explicit node list to execute (a scoped subgraph,
                e.g. a selected node's upstream closure).  When None the whole
                graph runs.
        """
        # Never execute a half-deserialised graph.  Callers that load a
        # project schedule their preview after ``_is_loading`` is cleared.
        if self._is_loading:
            return
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage("Computation already in progress...")
            return

        # Keep UI responsive during optimization (don't disable)
        # self.graph.widget.setEnabled(False)  # Removed for real-time viz
        # self.toolbar.setEnabled(False)  # Removed for real-time viz

        if skip_simulation:
            self.statusBar().showMessage("Updating design preview...")
        else:
            self.statusBar().showMessage(
                "Computing... (watch 3D viewer for live updates)"
            )
            self._last_topopt_preview_payload = None
            self.timeline.add_event("Graph execution started (Full)")

        # Capture the list of nodes on the MAIN THREAD.  Do not rewrite TopOpt
        # solver settings here; saved studies and explicit user edits must run
        # as-authored. Defaults remain available from the TopOpt property panel.
        all_nodes_snapshot = (
            list(nodes) if nodes is not None else list(self.graph.all_nodes())
        )
        has_topopt_run = False
        for node in all_nodes_snapshot:
            if getattr(node, "__identifier__", "") == "com.cad.sim.topopt_voxel":
                has_topopt_run = True
                if not skip_simulation:
                    message = self._topopt_preflight_error(node)
                    if message:
                        try:
                            node.set_error(message)
                        except Exception:
                            logger.debug("Optional UI operation failed.", exc_info=True)
                        self.statusBar().showMessage(message)
                        self.timeline.add_event(message)
                        QtWidgets.QMessageBox.warning(
                            self, "Topology Opt Setup", message
                        )
                        return
        self._prefer_topopt_after_run = bool(has_topopt_run and not skip_simulation)

        # Initialize worker with skip_simulation parameter
        self.worker = GraphExecutionWorker(
            all_nodes_snapshot, skip_simulation=skip_simulation, parent=self
        )

        self.worker.computation_finished.connect(self._on_execution_finished)
        self.worker.computation_cancelled.connect(self._on_execution_cancelled)
        self.worker.computation_error.connect(self._on_execution_error)
        # Connect optimization step for real-time visualization
        self.worker.optimization_step.connect(self._on_optimization_step)
        self.worker.start()

    def _cancel_execution(self):
        worker = getattr(self, "worker", None)
        if worker is None or not worker.isRunning():
            self.statusBar().showMessage("No computation is running")
            return
        worker.cancel()
        self.statusBar().showMessage(
            "Stopping computation at the next safe iteration..."
        )
        self.timeline.add_event("Computation stop requested")

    @QtCore.Slot(bool)
    @QtCore.Slot()  # Allow calling without arguments (default=False)
    def execute_graph(self, skip_simulation=False):
        """Public alias for _execute_graph, allowed to be called by external agents."""
        self._execute_graph(skip_simulation)

    def _on_optimization_step(self, mesh, densities, step, total):
        """Update the 3D viewer with current optimization state (real-time viz)."""
        try:
            import numpy as np

            now = time.monotonic()
            stage = mesh.get("stage") if isinstance(mesh, dict) else None
            vol_frac = None
            if isinstance(mesh, dict):
                try:
                    vol_frac = float(mesh.get("final_vol_frac"))
                except Exception:
                    vol_frac = None
            if vol_frac is None or not np.isfinite(vol_frac):
                vol_frac = float(np.mean(densities))
            is_final_step = (step + 1) >= total
            if (
                not stage
                and not is_final_step
                and (now - self._last_preview_update_time) < 0.1
            ):
                return

            self._last_preview_update_time = now
            if stage:
                self.statusBar().showMessage(f"TopOpt: {stage} (Vol: {vol_frac:.1%})")
            else:
                self.statusBar().showMessage(
                    f"TopOpt: Iteration {step + 1}/{total} (Vol: {vol_frac:.1%})"
                )

            if isinstance(mesh, dict) and mesh.get("type") == "topopt_voxel":
                result = dict(mesh)
                result["_preview"] = True
                self._last_topopt_preview_payload = result
                try:
                    for candidate in self.graph.all_nodes():
                        if (
                            getattr(candidate, "__identifier__", "")
                            == "com.cad.sim.topopt_voxel"
                        ):
                            self._last_rendered_node = candidate
                            break
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)
                self.viewer.render_simulation(result)
                return

        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
