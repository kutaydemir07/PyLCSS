# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""WorkbenchExportMixin behavior for the Design Studio workbench."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import numpy as np
from PySide6 import QtWidgets


logger = logging.getLogger(__name__)

__all__ = ["WorkbenchExportMixin"]


class WorkbenchExportMixin:
    def _get_exportable_result_node(self):
        """Return the best candidate node with cached simulation results."""
        candidates = []
        selected = next(iter(self.graph.selected_nodes()), None)
        if selected is not None:
            candidates.append(selected)

        current = getattr(self.properties, "current_node", None)
        if current is not None and current not in candidates:
            candidates.append(current)

        last = getattr(self, "_last_rendered_node", None)
        if last is not None and last not in candidates:
            candidates.append(last)

        for node in reversed(list(self.graph.all_nodes())):
            if node not in candidates:
                candidates.append(node)

        for node in candidates:
            result = getattr(node, "_last_result", None)
            if not isinstance(result, dict):
                continue
            rtype = result.get("type")
            # Voxel topology-opt results carry 'density'/'recovered_shape'
            # instead of a scikit-fem 'mesh', so accept them explicitly.
            if rtype == "topopt_voxel":
                return node
            if result.get("mesh") is not None and rtype in {"fea", "crash"}:
                return node
        return None

    def _find_renderable_simulation_node(self):
        """Return a graph node whose cached result is a renderable simulation
        (topopt density, FEA field, impact frames, or a recovered shape).

        Prefers the optimization / FEA *producers* over downstream consumers
        so the topopt result stays visible after a run even when CAD / export
        nodes are wired after it.
        """
        sim_nodes = []
        try:
            all_nodes = list(self.graph.all_nodes())
        except Exception:
            return None
        for node in all_nodes:
            result = getattr(node, "_last_result", None)
            if self._is_simulation_render_result(result):
                sim_nodes.append(node)

        def _rank(node):
            result = getattr(node, "_last_result", None)
            rtype = result.get("type") if isinstance(result, dict) else None
            if rtype in ("topopt_voxel",):
                return 0
            if rtype == "fea":
                return 1
            if rtype == "crash":
                return 2
            if rtype == "remesh":
                return 3
            if isinstance(result, dict) and "vertices" in result and "faces" in result:
                return 4
            return 3

        sim_nodes.sort(key=_rank)
        return sim_nodes[0] if sim_nodes else None

    @staticmethod
    def _build_topopt_export_payload(node, result):
        """Create portable JSON/HDF5 data for a structured voxel result."""

        density = np.asarray(result.get("density"), dtype=float)
        if density.ndim != 3 or density.size == 0:
            raise ValueError("The topology result has no 3-D density field to export.")

        node_name = getattr(node, "name", None)
        if callable(node_name):
            node_name = node_name()
        node_name = (
            node_name or getattr(node, "NODE_NAME", None) or node.__class__.__name__
        )
        metadata = {
            "node_name": str(node_name),
            "node_class": node.__class__.__name__,
            "simulation_type": "topopt_voxel",
            "visualization_mode": str(result.get("visualization_mode", "")),
            "exported_at": datetime.now().isoformat(),
            "cell_type": "voxel",
            "cell_count": int(density.size),
            "grid_shape": [int(v) for v in density.shape],
        }

        summary = {}
        for key in (
            "target_vol_frac",
            "final_vol_frac",
            "bounding_vol_frac",
            "compliance",
            "stress_pnorm",
            "volume",
            "mass",
            "total_volume",
            "density_equivalent_volume",
            "density_equivalent_mass",
            "recovered_design_volume",
            "recovered_design_mass",
            "recovered_assembly_volume",
            "recovered_assembly_mass",
            "assembly_hardware_volume",
            "recovery_volume_delta_pct",
            "iterations",
            "max_iterations",
            "density_cutoff",
        ):
            value = result.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                summary[key] = float(value)
        summary["converged"] = bool(result.get("converged"))
        summary["message"] = str(result.get("message") or "")
        summary["design_goal"] = str(result.get("design_goal") or "")

        fields = {"density": density.tolist()}
        hdf5_datasets = {"voxel/density": density}
        for key in ("design_density", "design_domain"):
            value = result.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.shape == density.shape:
                fields[key] = arr.tolist()
                hdf5_datasets[f"voxel/{key}"] = arr

        history = {}
        for key in (
            "compliance_history",
            "change_history",
            "stress_history",
            "objective_history",
        ):
            value = result.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float)
            history[key] = arr.tolist()
            hdf5_datasets[f"history/{key}"] = arr

        json_payload = {
            "metadata": metadata,
            "summary": summary,
            "bounds": result.get("bounds"),
            "voxel_fields": fields,
            "history": history,
        }
        recovered = result.get("recovered_shape")
        if (
            isinstance(recovered, dict)
            and recovered.get("vertices") is not None
            and recovered.get("faces") is not None
        ):
            vertices = np.asarray(recovered["vertices"], dtype=float)
            faces = np.asarray(recovered["faces"], dtype=int)
            json_payload["recovered_shape"] = {
                "vertices": vertices.tolist(),
                "faces": faces.tolist(),
            }
            hdf5_datasets["recovered_shape/vertices"] = vertices
            hdf5_datasets["recovered_shape/faces"] = faces
        return json_payload, hdf5_datasets, metadata

    def _build_simulation_export_payload(self, node):
        """Create portable JSON/HDF5 payloads from a cached simulation result."""

        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict):
            raise ValueError("The selected node has no exportable simulation result.")
        if result.get("type") == "topopt_voxel":
            return self._build_topopt_export_payload(node, result)
        if result.get("mesh") is None:
            raise ValueError("The selected node has no exportable simulation mesh.")

        mesh = result["mesh"]
        if not hasattr(mesh, "p") or not hasattr(mesh, "t"):
            raise ValueError(
                "Only scikit-fem style simulation meshes are supported for export."
            )

        points = np.asarray(mesh.p.T, dtype=float)
        if points.ndim != 2:
            raise ValueError("Invalid mesh point array.")
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points))])

        from pylcss.solver_backends.mesh import tet10_connectivity

        quadratic_connectivity = tet10_connectivity(mesh)
        connectivity = (
            quadratic_connectivity if quadratic_connectivity is not None else mesh.t
        )
        cells = np.asarray(connectivity.T, dtype=int)
        if cells.ndim != 2:
            raise ValueError("Invalid mesh connectivity array.")

        n_points = points.shape[0]
        n_cells = cells.shape[0]
        nodes_per_cell = cells.shape[1]
        cell_type_map = {
            2: "line",
            3: "triangle",
            4: "tetra",
            8: "hexahedron",
            10: "tetra10",
        }
        cell_type = cell_type_map.get(nodes_per_cell, f"{nodes_per_cell}-node")

        def _as_numeric_array(value):
            if value is None:
                return None
            arr = np.asarray(value)
            if arr.size == 0 or arr.dtype == object:
                return None
            return arr

        point_data = {}
        cell_data = {}
        history = {}
        recovered_shape = None

        displacement = _as_numeric_array(result.get("displacement"))
        displacement_vec = None
        if (
            displacement is not None
            and displacement.ndim == 1
            and displacement.size == 3 * n_points
        ):
            displacement_vec = displacement.reshape(n_points, 3)
        elif (
            displacement is not None
            and displacement.ndim == 2
            and displacement.shape[0] == n_points
        ):
            if displacement.shape[1] == 2:
                displacement_vec = np.column_stack([displacement, np.zeros(n_points)])
            elif displacement.shape[1] >= 3:
                displacement_vec = displacement[:, :3]

        if displacement_vec is not None:
            point_data["displacement"] = displacement_vec
            point_data["displacement_magnitude"] = np.linalg.norm(
                displacement_vec, axis=1
            )

        stress = _as_numeric_array(result.get("stress"))
        if stress is not None:
            if stress.ndim == 1 and stress.size == n_points:
                point_data["stress"] = stress
            elif stress.ndim == 1 and stress.size == n_cells:
                cell_data["stress"] = stress

        density = _as_numeric_array(result.get("density"))
        if density is not None:
            if density.ndim == 1 and density.size == n_cells:
                cell_data["density"] = density
            elif density.ndim == 1 and density.size == n_points:
                point_data["density"] = density

        design_density = _as_numeric_array(result.get("design_density"))
        if (
            design_density is not None
            and design_density.ndim == 1
            and design_density.size == n_cells
        ):
            cell_data["design_density"] = design_density

        element_stress = _as_numeric_array(result.get("element_stress"))
        if (
            element_stress is not None
            and element_stress.ndim == 1
            and element_stress.size == n_cells
        ):
            cell_data["element_stress"] = element_stress

        plastic_strain = _as_numeric_array(result.get("plastic_strain"))
        if (
            plastic_strain is not None
            and plastic_strain.ndim == 1
            and plastic_strain.size == n_cells
        ):
            cell_data["plastic_strain"] = plastic_strain

        failed_elements = _as_numeric_array(result.get("failed_elements"))
        if (
            failed_elements is not None
            and failed_elements.ndim == 1
            and failed_elements.size == n_cells
        ):
            cell_data["failed_elements"] = failed_elements.astype(np.int8)

        for key in (
            "time",
            "energy_kinetic",
            "energy_strain",
            "energy_plastic",
            "energy_balance",
        ):
            arr = _as_numeric_array(result.get(key))
            if arr is not None:
                history[key] = arr

        topopt_shape = result.get("recovered_shape")
        if isinstance(topopt_shape, dict):
            vertices = _as_numeric_array(topopt_shape.get("vertices"))
            faces = _as_numeric_array(topopt_shape.get("faces"))
            if vertices is not None and faces is not None:
                recovered_shape = {
                    "vertices": vertices,
                    "faces": faces,
                }

        node_name = getattr(node, "name", None)
        if callable(node_name):
            node_name = node_name()
        if not node_name:
            node_name = getattr(node, "NODE_NAME", None) or node.__class__.__name__

        metadata = {
            "node_name": str(node_name),
            "node_class": node.__class__.__name__,
            "simulation_type": str(result.get("type", "unknown")),
            "visualization_mode": str(result.get("visualization_mode", "")),
            "exported_at": datetime.now().isoformat(),
            "cell_type": cell_type,
            "point_count": int(n_points),
            "cell_count": int(n_cells),
        }

        summary = {}
        for key in (
            "peak_displacement",
            "peak_stress_nodal",
            "strain_energy",
            "compliance",
            "volume",
            "mass",
            "peak_stress",
            "absorbed_energy",
            "n_failed",
            "energy_balance_max_error",
            "max_stress_gauss",
            "deformation_scale",
            "density_cutoff",
        ):
            value = result.get(key)
            if isinstance(value, (int, float)):
                summary[key] = float(value)

        json_payload = {
            "metadata": metadata,
            "summary": summary,
            "mesh": {
                "points": points.tolist(),
                "cell_type": cell_type,
                "cells": cells.tolist(),
            },
            "point_data": {
                name: np.asarray(values).tolist() for name, values in point_data.items()
            },
            "cell_data": {
                name: np.asarray(values).tolist() for name, values in cell_data.items()
            },
            "history": {
                name: np.asarray(values).tolist() for name, values in history.items()
            },
        }

        if recovered_shape is not None:
            json_payload["recovered_shape"] = {
                "vertices": recovered_shape["vertices"].tolist(),
                "faces": recovered_shape["faces"].tolist(),
            }

        hdf5_datasets = {
            "mesh/points": points,
            "mesh/cells": cells,
        }
        for name, values in point_data.items():
            hdf5_datasets[f"point_data/{name}"] = np.asarray(values)
        for name, values in cell_data.items():
            hdf5_datasets[f"cell_data/{name}"] = np.asarray(values)
        for name, values in history.items():
            hdf5_datasets[f"history/{name}"] = np.asarray(values)
        if recovered_shape is not None:
            hdf5_datasets["recovered_shape/vertices"] = recovered_shape["vertices"]
            hdf5_datasets["recovered_shape/faces"] = recovered_shape["faces"]

        return json_payload, hdf5_datasets, metadata

    def _export_simulation_results(self, node=None):
        """Save cached FEA/TopOpt/impact results without re-running a study.

        ``node`` is optional; the main toolbar normally uses the active or last
        terminal result. QAction may supply its checked-state boolean, which is
        treated as no explicit node.
        """
        if not self._ensure_idle_for_io("exporting simulation results"):
            return

        if isinstance(node, bool):
            node = None
        node = node or self._get_exportable_result_node()
        if node is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Results",
                "Run or select an FEA, topology-optimization, or impact node before exporting results.",
            )
            return

        try:
            json_payload, hdf5_datasets, metadata = (
                self._build_simulation_export_payload(node)
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))
            return

        node_name = metadata.get("node_name", "simulation_results")
        safe_name = (
            "".join(
                ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in node_name
            ).strip("_")
            or "simulation_results"
        )
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Simulation Results",
            f"{safe_name}.json",
            "JSON Files (*.json);;HDF5 Files (*.h5)",
        )
        if not fname:
            return

        ext = os.path.splitext(fname)[1].lower()
        if not ext:
            fname += ".json"
            ext = ".json"

        try:
            from pylcss.io_manager.data_io import DataExporter

            if ext == ".json":
                DataExporter.to_json(fname, json_payload)
            elif ext == ".h5":
                attrs = {
                    k: v
                    for k, v in metadata.items()
                    if isinstance(v, (str, int, float, bool))
                }
                for key, value in json_payload.get("summary", {}).items():
                    attrs[f"summary_{key}"] = value
                DataExporter.to_hdf5(fname, hdf5_datasets, attrs=attrs)
            else:
                raise ValueError(f"Unsupported export format: {ext}")

            self.timeline.add_event(f"Exported simulation results: {fname}")
            self.statusBar().showMessage(f"Exported results: {fname}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))
