# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""WorkbenchTopologyMixin behavior for the Design Studio workbench."""

from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtCore, QtWidgets

from pylcss.design_studio.topology_optimization.integration.study_identity import (
    DENSITY_STUDY_IDENTIFIERS,
    is_density_study_node,
)

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

    @staticmethod
    def _connected_input_source_nodes(node, port_name):
        """Return the nodes feeding one input port without executing the graph."""
        try:
            port = node.get_input(port_name)
            connected = list(port.connected_ports()) if port else []
        except Exception:
            connected = []
        sources = []
        for source_port in connected:
            try:
                sources.append(source_port.node())
            except Exception:
                continue
        return sources

    @classmethod
    def _topopt_has_boundary_connection(
        cls,
        node,
        direct_port,
        operating_case_port,
    ):
        """Recognize both global and operating-case-local TopOpt boundaries."""
        if cls._port_has_connections(node, direct_port):
            return True
        for case_node in cls._connected_input_source_nodes(node, "load_cases"):
            if cls._port_has_connections(case_node, operating_case_port):
                return True
        return False

    def _topopt_preflight_error(self, node):
        if not is_density_study_node(node):
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
        elif goal in {
            "thermo-mechanical",
            "coupled structural + thermal",
        }:
            physics = "thermo-mechanical"
        elif goal in {
            "lightweight stiffness",
            "minimum mass under stress",
        }:
            physics = "structural"

        if physics in {"structural", "thermo-mechanical"}:
            # Joints are optional and self-declaring: a study that has them is
            # multi-body, one that does not is a single body. Neither case is
            # announced through the design goal any more.
            if not self._topopt_has_boundary_connection(
                node,
                "supports",
                "supports",
            ):
                return (
                    "Structural TopOpt needs a Support, connected "
                    "directly or through a Topology Load Case and scoped "
                    "to a selected design-domain interface."
                )
            if not self._topopt_has_boundary_connection(
                node,
                "loads",
                "loads",
            ):
                return (
                    "Structural TopOpt needs a Force, connected "
                    "directly or through a Topology Load Case and scoped "
                    "to a selected design-domain interface."
                )

        if physics in {"thermal", "thermo-mechanical"}:
            if not self._port_has_connections(node, "thermal_sinks"):
                return (
                    "Thermal TopOpt needs a connected Reference Temperature "
                    "scoped to a selected design-domain interface."
                )
            if not self._port_has_connections(node, "thermal_loads"):
                return (
                    "Thermal TopOpt needs a connected Topology Heat Load scoped to a "
                    "selected design-domain interface."
                )

        return None

    @staticmethod
    def _topopt_cached_domain_value(node):
        sources = WorkbenchTopologyMixin._connected_input_source_nodes(
            node,
            "design_domain",
        )
        if not sources:
            return None
        source = sources[0]
        value = getattr(source, "_last_result", None)
        if isinstance(value, dict):
            try:
                port = node.get_input("design_domain")
                source_port = list(port.connected_ports())[0]
                output_name = source_port.name()
            except Exception:
                output_name = ""
            if output_name and output_name in value:
                value = value[output_name]
            elif "shape" in value:
                value = value["shape"]
        return value

    def _find_cached_topopt_domain(self):
        """Return a TopOpt node and its computed design-domain preview."""
        try:
            nodes = list(self.graph.all_nodes())
        except Exception:
            return None, None

        selected = next(iter(self.graph.selected_nodes()), None)
        candidates = []
        if selected is not None and is_density_study_node(selected):
            candidates.append(selected)
        candidates.extend(
            node
            for node in nodes
            if node not in candidates and is_density_study_node(node)
        )
        for node in candidates:
            value = self._topopt_cached_domain_value(node)
            if self._is_renderable_result(value):
                return node, value
        return None, None

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
        # Keep automatic grids on the geometric-multigrid path. Odd axes make
        # the solver fall back to sparse LU, which is needlessly slow and
        # memory-heavy for normal 3-D studies.
        dims += dims % 2
        while int(np.prod(dims)) > max_cells:
            axis = int(np.argmax(dims))
            if int(dims[axis]) <= min_axis:
                break
            dims[axis] = max(min_axis, int(dims[axis]) - 2)
        return [int(v) for v in dims]

    def _apply_topopt_industrial_defaults(self, node):
        if not is_density_study_node(node):
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
            else "Coupled Structural + Thermal"
            if goal in {
                "thermo-mechanical",
                "coupled structural + thermal",
            }
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
            "density_cutoff": 0.30,
            "visualization": "Density",
            "optimizer": (
                "GCMMA"
                if stress_enabled or stress_goal
                else "MMA"
                if goal
                in {
                    "thermo-mechanical",
                    "coupled structural + thermal",
                }
                else "Auto"
            ),
            "physics_mode": physics_mode,
            # `load_aggregation` is the user's call about the load envelope and
            # is deliberately not reset here.
            "max_iter": 100,
            "tol": 0.005,
            "convergence_patience": 5,
            "print_ready_mesh": False,
            "mesh_decimate_ratio": 1.0,
            "surface_recovery_method": "Volume-Preserving SDF (VTK)",
            "exclusion_scope": "All Loads and Supports",
            "exclusion_thickness_mode": "Program Controlled",
            "exclusion_thickness_mm": 2.0,
        }
        if node.has_property("lattice_settings_mode"):
            # A lattice study's cell family is its definition and is never
            # reset here; only the coupled dimensions return to Guided, which
            # re-derives them from the grid this call just sized.
            settings.update(
                {
                    "lattice_settings_mode": "Guided",
                    "structure_cell_size_voxels": 8.0,
                    "structure_member_thickness_voxels": 1.0,
                    "structure_skin_thickness_voxels": 0.75,
                    "lattice_porosity": "Conservative",
                }
            )
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
            *DENSITY_STUDY_IDENTIFIERS,
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
                    "the FEA, impact, or topology solver you want to run."
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
            self._last_rendered_geom = None
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

    def _execute_graph(self, skip_simulation=False, nodes=None, skip_meshing=False):
        """Start graph execution in a background thread.

        Args:
            skip_simulation: If True, skip FEA/TopOpt nodes (for auto-update mode)
            nodes: Optional explicit node list to execute (a scoped subgraph,
                e.g. a selected node's upstream closure).  When None the whole
                graph runs.
            skip_meshing: If True, also skip mesh/remesh nodes.  Used for the
                refresh that follows opening a project, where meshing a whole
                model cold costs as much as the solve the preview exists to
                avoid.  Ignored unless ``skip_simulation`` is also set.
        """
        # Never execute a half-deserialised graph.  Callers that load a
        # project schedule their preview after ``_is_loading`` is cleared.
        if self._is_loading:
            return
        if not self._confirm_project_code_execution():
            return
        if self.worker and self.worker.isRunning():
            # Do not discard the request. Property edits, connection changes
            # and node selection all schedule a refresh, and dropping the ones
            # that land during an in-flight run left the viewer and the
            # inspector showing stale geometry until something else happened
            # to trigger another run.
            self._queue_execute(
                skip_simulation=skip_simulation,
                skip_meshing=skip_meshing,
                nodes=nodes,
            )
            if skip_simulation:
                self.statusBar().showMessage("Preview queued")
            else:
                self.statusBar().showMessage("Solve queued")
            return

        # A queued full solve outranks the preview that is about to start.
        # Clearing the queue unconditionally is what lost an explicit Run: any
        # auto-refresh (property edit, selection, connection change) landing
        # between the click and the queued start discarded it silently, and the
        # only way to notice was that nothing happened until Run was pressed a
        # second time.
        pending = self._pending_execute
        self._pending_execute = None
        if pending is not None:
            pending_skip, pending_skip_meshing, pending_nodes = pending
            outranks_by_depth = bool(skip_simulation) and not pending_skip
            # A queued request that still wants a mesh must survive the
            # geometry-only refresh that opening a project schedules.
            outranks_by_meshing = bool(skip_meshing) and not pending_skip_meshing
            outranks_by_scope = pending_nodes is None and nodes is not None
            if outranks_by_depth or outranks_by_meshing or outranks_by_scope:
                self._pending_execute = pending

        # Keep UI responsive during optimization (don't disable)
        # self.graph.widget.setEnabled(False)  # Removed for real-time viz
        # self.toolbar.setEnabled(False)  # Removed for real-time viz

        if skip_simulation:
            self.statusBar().showMessage(
                "Updating geometry preview (mesh runs on Solve)..."
                if skip_meshing
                else "Updating preview..."
            )
        else:
            self.statusBar().showMessage("Solving...")
            self._set_solve_progress("Solving...")
            self._last_topopt_preview_payload = None
            self.timeline.add_event("Solve started")

        # Capture the list of nodes on the MAIN THREAD.  Do not rewrite TopOpt
        # solver settings here; saved studies and explicit user edits must run
        # as-authored. Defaults remain available from the TopOpt property panel.
        all_nodes_snapshot = (
            list(nodes) if nodes is not None else list(self.graph.all_nodes())
        )
        has_topopt_run = False
        for node in all_nodes_snapshot:
            if is_density_study_node(node):
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
            all_nodes_snapshot,
            skip_simulation=skip_simulation,
            skip_meshing=skip_meshing,
            parent=self,
        )

        self.worker.computation_finished.connect(self._on_execution_finished)
        self.worker.computation_cancelled.connect(self._on_execution_cancelled)
        self.worker.computation_error.connect(self._on_execution_error)
        # One pre-solve preview plus lightweight iteration status; VTK is not
        # rebuilt during iterations.
        self.worker.optimization_step.connect(self._on_optimization_step)
        self.worker.start()

    def _queue_execute(self, *, skip_simulation, skip_meshing, nodes):
        """Merge an execution request that arrived while a run was in flight.

        Only one request is kept, and merging always widens it: a full solve
        never degrades into a preview, a meshing preview never degrades into a
        geometry-only one, and a whole-graph run never narrows to a scoped
        subgraph. That keeps an explicit Run pressed during an auto-refresh
        from being silently downgraded.
        """
        pending_skip, pending_skip_meshing, pending_nodes = (
            self._pending_execute or (True, True, ())
        )
        merged_skip = bool(skip_simulation) and bool(pending_skip)
        merged_skip_meshing = bool(skip_meshing) and bool(pending_skip_meshing)
        if nodes is None or pending_nodes is None:
            merged_nodes = None
        else:
            merged = list(pending_nodes)
            seen = {id(node) for node in merged}
            for node in nodes:
                if id(node) not in seen:
                    seen.add(id(node))
                    merged.append(node)
            merged_nodes = merged
        self._pending_execute = (merged_skip, merged_skip_meshing, merged_nodes)

    def _start_pending_execute(self):
        """Run the request that was queued during the previous execution."""
        pending = self._pending_execute
        self._pending_execute = None
        if pending is None or self._is_loading:
            return
        skip_simulation, skip_meshing, nodes = pending
        # An empty scoped list means "nothing left to run", not "whole graph".
        if nodes is not None and not nodes:
            return
        self._execute_graph(
            skip_simulation=skip_simulation,
            nodes=nodes,
            skip_meshing=skip_meshing,
        )

    def _cancel_execution(self):
        worker = getattr(self, "worker", None)
        if worker is None or not worker.isRunning():
            self.statusBar().showMessage("Nothing running")
            return
        # Stopping is an explicit request to end work, so drop anything that
        # was queued behind the running solve instead of starting it next.
        self._pending_execute = None
        worker.cancel()
        self.statusBar().showMessage("Stopping...")
        self.timeline.add_event("Stopping solve")

    @QtCore.Slot(bool)
    @QtCore.Slot()  # Allow calling without arguments (default=False)
    def execute_graph(self, skip_simulation=False):
        """Public alias for _execute_graph, allowed to be called by external agents."""
        self._execute_graph(skip_simulation)

    def _on_optimization_step(self, mesh, densities, step, total):
        """Render TopOpt milestones while status-only updates leave the actor intact."""
        try:
            if isinstance(mesh, dict) and mesh.get("status_only"):
                stage = str(mesh.get("stage") or "").strip()
                if stage:
                    self._set_solve_progress(stage)
                    self.statusBar().showMessage(f"Topology: {stage}")
                    return
                objective = mesh.get("objective")
                change = mesh.get("design_change")
                details = []
                if objective is not None:
                    details.append(f"objective {float(objective):.4g}")
                if change is not None:
                    details.append(f"change {float(change):.3g}")
                suffix = f" — {', '.join(details)}" if details else ""
                self._set_solve_progress(
                    f"Iteration {step + 1} of {total}",
                    value=step + 1,
                    maximum=total,
                )
                self.statusBar().showMessage(
                    f"Topology iteration {step + 1} of {total}{suffix}"
                )
                return

            stage = mesh.get("stage") if isinstance(mesh, dict) else None
            vol_frac = None
            if isinstance(mesh, dict):
                try:
                    vol_frac = float(mesh.get("final_vol_frac"))
                except Exception:
                    vol_frac = None
            if (
                vol_frac is None
                and densities is not None
                and np.asarray(densities).size
            ):
                vol_frac = float(np.mean(densities))
            if vol_frac is None or not np.isfinite(vol_frac):
                vol_frac = 0.0
            if stage:
                target = (
                    float(mesh.get("target_vol_frac"))
                    if isinstance(mesh, dict)
                    and mesh.get("target_vol_frac") is not None
                    else None
                )
                target_text = (
                    f" (target material: {target:.1%})"
                    if target is not None and np.isfinite(target)
                    else ""
                )
                self._set_solve_progress(str(stage))
                self.statusBar().showMessage(f"Topology: {stage}{target_text}")
            else:
                self._set_solve_progress(
                    f"Iteration {step + 1} of {total}",
                    value=step + 1,
                    maximum=total,
                )
                self.statusBar().showMessage(
                    f"Topology iteration {step + 1} of {total} "
                    f"· material {vol_frac:.1%}"
                )

            if isinstance(mesh, dict) and mesh.get("type") == "topopt_voxel":
                result = dict(mesh)
                result["_preview"] = True
                self._last_topopt_preview_payload = result
                candidate = None
                try:
                    for candidate in self.graph.all_nodes():
                        if is_density_study_node(candidate):
                            self._last_rendered_node = candidate
                            break
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)
                # The honest pre-solve view is the exact CAD design domain.
                # A thresholded uniform starting density adds no engineering
                # information and makes coarse voxel boundaries look like a
                # fabricated topology result. Fall back to the closed voxel
                # preview only when the upstream CAD value is unavailable.
                if (
                    result.get("preview_kind") == "design_domain"
                    and candidate is not None
                ):
                    domain_value = self._topopt_cached_domain_value(candidate)
                    if self._is_renderable_result(domain_value):
                        self._render_result_in_viewer(domain_value)
                        return
                self.viewer.render_simulation(result)
                return

        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
