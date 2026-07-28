# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Background execution and export workers for Design Studio."""

from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtCore

logger = logging.getLogger(__name__)

__all__ = ["GraphExecutionWorker", "TopOptStepExportWorker"]


class GraphExecutionWorker(QtCore.QThread):
    """Background worker to run the node graph without freezing the UI."""

    computation_finished = QtCore.Signal(object)  # Emits results dict
    computation_cancelled = QtCore.Signal(object)  # Emits safe partial results
    computation_error = QtCore.Signal(str)
    optimization_step = QtCore.Signal(
        object, object, int, int
    )  # mesh, densities, step, total

    def __init__(self, nodes, skip_simulation=False, parent=None):
        super().__init__(parent)
        self.nodes = nodes
        self.skip_simulation = skip_simulation
        self._is_running = False

    def run(self):
        self._is_running = True
        try:
            from pylcss.design_studio.engine import (
                GraphExecutionCancelled,
                execute_graph,
            )

            # Callback for real-time updates
            def progress_cb(mesh, densities, step, total):
                if self._is_running:
                    self.optimization_step.emit(mesh, densities, step, total)

            # Pass skip_simulation and callback to engine
            results = execute_graph(
                self.nodes,
                skip_simulation=self.skip_simulation,
                cancel_callback=lambda: not self._is_running,
                progress_callback=progress_cb,
            )

            if self._is_running:
                self.computation_finished.emit(results)
            else:
                self.computation_cancelled.emit(results)
        except GraphExecutionCancelled as exc:
            self.computation_cancelled.emit(exc.results)
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.computation_error.emit(str(e))
        finally:
            self._is_running = False

    def cancel(self):
        """Ask long-running simulation nodes to stop cleanly."""
        self._is_running = False
        self.requestInterruption()
        for node in self.nodes:
            request_stop = getattr(node, "request_stop", None)
            if callable(request_stop):
                request_stop()


def _external_write_cad_step(payload, path):
    import cadquery as cq
    from pylcss.design_studio.topology_optimization.geometry.cad_reconstruction import (
        reconstruct_topopt_cad,
    )

    shape = reconstruct_topopt_cad(
        payload,
        source_geometry="Recovered Shape",
        sew_tolerance=1e-4,
        max_faces=1500,
    )
    cq.exporters.export(shape, str(path), exportType="STEP")
    return True


class TopOptStepExportWorker(QtCore.QThread):
    """Background worker for TopOpt STEP export."""

    export_finished = QtCore.Signal(str)
    export_error = QtCore.Signal(str)

    def __init__(
        self,
        topo_output,
        path,
        *,
        density_cutoff=0.45,
        extrusion_axis="none",
        passive_regions=None,
        parent=None,
    ):
        super().__init__(parent)
        self.topo_output = dict(topo_output or {})
        self.path = str(path)
        self.density_cutoff = float(density_cutoff or 0.45)
        self.extrusion_axis = str(extrusion_axis or "none").strip().lower()
        self.passive_regions = dict(passive_regions or {})

    @staticmethod
    def _bounds_tuple(bounds_payload):
        if (
            isinstance(bounds_payload, dict)
            and "min" in bounds_payload
            and "max" in bounds_payload
        ):
            mins = np.asarray(bounds_payload["min"], dtype=float)
            maxs = np.asarray(bounds_payload["max"], dtype=float)
            if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
                return mins[:3], maxs[:3]
        return None

    def run(self):
        try:
            payload = dict(self.topo_output)
            payload["density_cutoff"] = self.density_cutoff
            if self.passive_regions:
                payload["passive_regions"] = self.passive_regions
            else:
                payload.setdefault("passive_regions", {})
            payload["extrusion_axis"] = self.extrusion_axis

            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_external_write_cad_step, payload, self.path)
                future.result()

            self.export_finished.emit(self.path)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.export_error.emit(str(exc))
