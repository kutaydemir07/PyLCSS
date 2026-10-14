# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""CalculiX FRD-to-PyLCSS result normalization."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from pylcss.cae.solver_backends.frd_reader import read_frd
from pylcss.cae.solver_backends.base import SolverBackendError
from pylcss.cae.solver_backends.mesh import (
    load_vector,
    mesh_to_tet4,
    tet10_connectivity,
)
from pylcss.cae.solver_backends.selection import (
    dict_geometries,
    nodes_matching_condition,
    nodes_matching_geometries,
    tet_face_sets_for_geometries,
)
from pylcss.cae.solver_backends.validation import (
    finite_float,
    nonnegative_float,
    positive_float,
)


logger = logging.getLogger(__name__)


def _compute_applied_forces(
    mesh: Any,
    loads: list[dict] | None,
    material: dict | None,
    total_volume: float,
) -> np.ndarray:
    """Compute resultant 3D vector of all external loads applied to the model."""
    applied_force = np.zeros(3, dtype=float)
    for index, load in enumerate(loads or [], start=1):
        if not isinstance(load, dict):
            raise SolverBackendError(f"Load {index} must be a dictionary.")
        ltype = str(load.get("type", "force") or "force").strip().lower()
        if ltype in ("force", "bearing_load", "bearing"):
            vector = load.get("vector")
            if vector is None:
                vector = [load.get(f"force_{axis}", 0.0) for axis in "xyz"]
            applied_force += load_vector({"vector": vector})
        elif ltype == "gravity":
            material_values = material or {}
            rho = positive_float(
                material_values.get("rho", material_values.get("density", 7.85e-9)),
                label="Material density",
            )
            mass = nonnegative_float(total_volume, label="Mesh volume") * rho
            accel = nonnegative_float(
                load.get("accel", 9810.0), label="Gravity acceleration"
            )
            direction = str(load.get("direction", "-Y") or "-Y").strip().upper()
            dir_map = {
                "-X": (-1.0, 0.0, 0.0),
                "+X": (1.0, 0.0, 0.0),
                "-Y": (0.0, -1.0, 0.0),
                "+Y": (0.0, 1.0, 0.0),
                "-Z": (0.0, 0.0, -1.0),
                "+Z": (0.0, 0.0, 1.0),
            }
            if direction not in dir_map:
                raise SolverBackendError(
                    f"Unsupported gravity direction: {direction!r}."
                )
            applied_force += mass * accel * np.asarray(dir_map[direction])
        elif ltype == "pressure":
            pressure_val = finite_float(
                load.get("pressure", load.get("magnitude", 0.0)),
                label=f"Pressure load {index} magnitude",
            )
            if abs(pressure_val) <= 1e-12:
                continue
            geoms = dict_geometries(load)
            faces = tet_face_sets_for_geometries(mesh, geoms) if geoms else []
            if not faces:
                raise SolverBackendError(
                    f"Pressure load {index} has no matched external mesh faces."
                )
            pts, cells = mesh_to_tet4(mesh, [])
            local_faces = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3))
            opp_nodes = (3, 2, 0, 1)
            for elem_id, face_id in faces:
                cell = cells[elem_id - 1]
                f_idx = face_id - 1
                tri = cell[list(local_faces[f_idx])]
                p0, p1, p2 = pts[tri]
                popp = pts[cell[opp_nodes[f_idx]]]
                c_face = (p0 + p1 + p2) / 3.0
                n_cross = np.cross(p1 - p0, p2 - p0)
                area = 0.5 * float(np.linalg.norm(n_cross))
                if area > 1e-16:
                    n_unit = n_cross / np.linalg.norm(n_cross)
                    if np.dot(n_unit, popp - c_face) < 0.0:
                        n_unit = -n_unit
                    applied_force += pressure_val * area * n_unit
    return applied_force


def _resolve_deformation_scale(
    requested: Any, auto_scale: float, warnings: list[str]
) -> float:
    """Resolve the visual-only deformation scale selected in the solver node."""
    if requested is None:
        return auto_scale
    if isinstance(requested, str):
        text = requested.strip().lower()
        if not text or text == "auto":
            return auto_scale
        text = text.rstrip("x").strip()
    else:
        text = requested
    try:
        value = float(text)
    except (TypeError, ValueError):
        warnings.append(f"Invalid deformation scale {requested!r}; using Auto.")
        return auto_scale
    if not math.isfinite(value) or value <= 0.0:
        warnings.append(
            f"Deformation scale must be finite and positive, got {value:g}; using Auto."
        )
        return auto_scale
    return value


def _ingest_frd_into_result(
    mesh: Any,
    frd_path: Path,
    visualization_mode: str,
    warnings: list[str],
    deformation_scale: Any = "Auto",
    analysis_type: str = "Linear",
    constraints: list[dict] | None = None,
    loads: list[dict] | None = None,
    material: dict | None = None,
) -> dict:
    """Read CalculiX ``.frd`` output and shape it into the viewer's result dict.

    The deck writes mesh nodes in order with 1-based ids (``i`` → ``i+1``).
    CCX's FRD preserves those ids, so we scatter the parsed displacement and
    stress back into the mesh's original ``[0..n_points-1]`` slot by
    ``id - 1``.  This is robust even if CCX drops or re-orders nodes.
    """
    parsed = read_frd(frd_path)

    n_points = int(np.asarray(mesh.p).shape[1])
    frd_node_ids = np.asarray(parsed["node_ids"], dtype=int)
    disp_xyz_frd = np.asarray(parsed["displacement"], dtype=float)
    vm_frd = np.asarray(parsed["von_mises"], dtype=float)
    stress_tensor_frd = np.asarray(parsed.get("stress", np.zeros((0, 6))), dtype=float)
    ener_frd = np.asarray(parsed.get("ener", np.zeros(0)), dtype=float)
    force_frd = np.asarray(parsed.get("nodal_force", np.zeros((0, 3))), dtype=float)
    for field_name, values in (
        ("node coordinates", parsed["nodes"]),
        ("displacement", disp_xyz_frd),
        ("stress", stress_tensor_frd),
        ("Von Mises stress", vm_frd),
        ("strain-energy density", ener_frd),
        ("nodal force", force_frd),
    ):
        if not np.all(np.isfinite(np.asarray(values, dtype=float))):
            raise ValueError(f"CalculiX FRD contains non-finite {field_name} values.")
    n_frd = frd_node_ids.size

    if n_frd != n_points:
        warnings.append(
            f"CalculiX FRD reports {n_frd} nodes; PyLCSS source mesh has "
            f"{n_points}.  Scattering by node id; any missing ids will render "
            "with zero displacement / zero stress."
        )

    # Scatter by id (CCX writes 1-based ids; mesh array is 0-based).
    disp_xyz: np.ndarray = np.zeros((n_points, 3), dtype=float)
    vm: np.ndarray = np.zeros(n_points, dtype=float)
    stress_tensor: np.ndarray = np.zeros((n_points, 6), dtype=float)
    ener_nodal: np.ndarray = np.zeros(n_points, dtype=float)
    nodal_force: np.ndarray = np.zeros((n_points, 3), dtype=float)
    if n_frd > 0:
        mesh_idx = frd_node_ids - 1
        valid = (mesh_idx >= 0) & (mesh_idx < n_points)
        if not valid.all():
            n_invalid = int((~valid).sum())
            warnings.append(
                f"{n_invalid} FRD node id(s) fall outside the source-mesh range; ignored."
            )
        idx_in = mesh_idx[valid]
        valid_src = np.where(valid)[0]
        if disp_xyz_frd.shape[0] >= valid.size:
            disp_xyz[idx_in] = disp_xyz_frd[valid_src]
        if vm_frd.size >= valid.size:
            vm[idx_in] = vm_frd[valid_src]
        if stress_tensor_frd.shape[0] >= valid.size:
            stress_tensor[idx_in] = stress_tensor_frd[valid_src]
        if ener_frd.size >= valid.size:
            ener_nodal[idx_in] = ener_frd[valid_src]
        if force_frd.shape[0] >= valid.size:
            nodal_force[idx_in] = force_frd[valid_src]

    constrained_nodes: set[int] = set()
    for constraint in constraints or []:
        geoms = dict_geometries(constraint)
        condition = str(constraint.get("condition", "") or "").strip()
        if geoms:
            selected = nodes_matching_geometries(mesh, geoms)
        elif condition:
            selected = nodes_matching_condition(
                mesh, condition, [], label="Reaction force"
            )
        else:
            continue
        constrained_nodes.update(int(v) for v in selected)
    constrained_node_ids = np.asarray(sorted(constrained_nodes), dtype=int)
    if constrained_node_ids.size:
        reaction_force = np.sum(nodal_force[constrained_node_ids], axis=0)
    else:
        reaction_force = np.zeros(3, dtype=float)

    # Viewer expects flat (3*N,) reshaping to (3, N) with order='F'.
    flat_disp: np.ndarray = np.zeros(3 * n_points, dtype=float)
    flat_disp[0::3] = disp_xyz[:, 0]
    flat_disp[1::3] = disp_xyz[:, 1]
    flat_disp[2::3] = disp_xyz[:, 2]

    max_disp = float(np.max(np.linalg.norm(disp_xyz, axis=1))) if disp_xyz.size else 0.0
    peak_vm = float(np.max(vm)) if vm.size else 0.0

    # Tet element volumes (used to integrate the nodal ENER density into the
    # total elastic strain energy and to project ENER back per-element for SIMP).
    quadratic_conn = tet10_connectivity(mesh)
    energy_conn = (
        quadratic_conn.T if quadratic_conn is not None else np.asarray(mesh.t).T[:, :4]
    ).astype(int)
    tets = energy_conn[:, :4]
    coords = np.asarray(mesh.p).T
    v0 = coords[tets[:, 0]]
    edges = np.stack(
        [coords[tets[:, 1]] - v0, coords[tets[:, 2]] - v0, coords[tets[:, 3]] - v0],
        axis=-1,
    )
    elem_vol = np.abs(np.linalg.det(edges)) / 6.0

    # Average every element node (4 for C3D4, 10 for C3D10).
    elem_ener = (
        ener_nodal[energy_conn].mean(axis=1)
        if ener_nodal.size
        else np.zeros(tets.shape[0])
    )

    # Total elastic strain energy ≈ Σ_e (ENER_e × V_e); compliance = 2 × SE for
    # linear elasticity (C = u·f = u·K·u = 2·SE).
    total_strain_energy = float(np.sum(elem_ener * elem_vol))
    # The 2*SE identity is valid only for a linear elastic load path.
    compliance = 2.0 * total_strain_energy if analysis_type == "Linear" else None

    logger.debug(
        "FEA Solver (external): FRD ingest frd_nodes=%d, mesh_nodes=%d, "
        "max|u|=%.4e, peak VM=%.4e, compliance=%.4e",
        n_frd,
        n_points,
        max_disp,
        peak_vm,
        compliance if compliance is not None else float("nan"),
    )

    # Auto deformation scale so deformed shape is visible without misleading.
    bbox = np.ptp(np.asarray(mesh.p), axis=1)
    char_len = max(float(np.max(bbox)), 1e-9)
    auto_scale = (0.05 * char_len / max_disp) if max_disp > 1e-9 else 1.0
    auto_scale = float(np.clip(auto_scale, 1.0, 200.0))
    resolved_scale = _resolve_deformation_scale(deformation_scale, auto_scale, warnings)

    total_volume = float(np.sum(elem_vol))

    # Static equilibrium verification
    applied_force = _compute_applied_forces(mesh, loads, material, total_volume)
    applied_magnitude = float(np.linalg.norm(applied_force))
    reaction_magnitude = float(np.linalg.norm(reaction_force))
    residual_force = applied_force + reaction_force
    residual_magnitude = float(np.linalg.norm(residual_force))
    ref_mag = max(applied_magnitude, reaction_magnitude, 1e-12)
    equilibrium_error_pct = (
        float((residual_magnitude / ref_mag) * 100.0) if ref_mag > 1e-9 else 0.0
    )
    equilibrium_balanced = bool(equilibrium_error_pct < 1.0)

    # Factor of Safety calculation
    yield_strength = 0.0
    if material is not None:
        try:
            yield_strength = float(
                material.get("yield_strength", material.get("yield_stress", 0.0)) or 0.0
            )
        except (TypeError, ValueError):
            yield_strength = 0.0

    safety_factor: np.ndarray | None = None
    min_safety_factor: float | None = None
    if yield_strength > 0.0 and vm.size:
        safety_factor = np.where(
            vm > 1e-12,
            np.clip(yield_strength / np.maximum(vm, 1e-12), 0.01, 100.0),
            100.0,
        )
        min_safety_factor = float(np.min(safety_factor))

    return {
        "type": "fea",
        "backend": "CalculiX",
        "mesh": mesh,
        "displacement": flat_disp,
        "stress": vm,
        "stress_tensor": stress_tensor,
        "safety_factor": safety_factor,
        "min_safety_factor": min_safety_factor,
        "yield_strength": yield_strength,
        "applied_force": applied_force,
        "applied_magnitude": applied_magnitude,
        "reaction_force": reaction_force,
        "reaction_magnitude": reaction_magnitude,
        "residual_force": residual_force,
        "residual_magnitude": residual_magnitude,
        "equilibrium_error_pct": equilibrium_error_pct,
        "equilibrium_balanced": equilibrium_balanced,
        "ener_nodal": ener_nodal,
        "nodal_force": nodal_force,
        "reaction_node_ids": constrained_node_ids,
        "element_ener": elem_ener,
        "element_volumes": elem_vol,
        "strain_energy": total_strain_energy,
        "compliance": compliance,
        "volume": total_volume,
        "peak_displacement": max_disp,
        "peak_stress_nodal": peak_vm,
        "stress_location": "nodal_extrapolated",
        "auto_deformation_scale": auto_scale,
        "deformation_scale": resolved_scale,
        "visualization_mode": visualization_mode,
        "frd_file": str(frd_path),
        "frd_steps": parsed.get("steps", []),
    }
