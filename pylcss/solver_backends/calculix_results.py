# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""CalculiX FRD-to-PyLCSS result normalization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from pylcss.solver_backends.frd_reader import read_frd
from pylcss.solver_backends.mesh import tet10_connectivity
from pylcss.solver_backends.selection import (
    dict_geometries,
    nodes_matching_condition,
    nodes_matching_geometries,
)


logger = logging.getLogger(__name__)


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
    if value <= 0.0:
        warnings.append(
            f"Deformation scale must be positive, got {value:g}; using Auto."
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

    return {
        "type": "fea",
        "backend": "CalculiX",
        "mesh": mesh,
        "displacement": flat_disp,
        "stress": vm,
        "stress_tensor": stress_tensor,
        "ener_nodal": ener_nodal,
        "nodal_force": nodal_force,
        "reaction_force": reaction_force,
        "reaction_magnitude": float(np.linalg.norm(reaction_force)),
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
