# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""CalculiX static-structural backend adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from pylcss.solver_backends.common import (
    ExternalRunConfig,
    SolverBackendError,
    dict_geometries,
    id_lines,
    load_vector,
    make_work_dir,
    mesh_to_tet4,
    nodes_matching_condition,
    nodes_matching_geometries,
    resolve_executable,
    run_process,
    tail,
    tet_face_sets_for_geometries,
)
from pylcss.solver_backends.frd_reader import read_frd


def _collect_calculix_failure_context(
    work_dir: Path, job_name: str, returncode: int, executable: str
) -> str:
    """Build a diagnostic block when ccx exits non-zero with little to no stdout.

    Reads the small log files CalculiX writes (.sta, .cvg, .dat, .log) and
    surfaces them.  Also detects the Windows 0xC0000135 / -1073741515 case
    where the .exe failed to load because of a missing DLL.
    """
    parts: List[str] = []
    parts.append(f"Exit code: {returncode} (0x{(returncode & 0xFFFFFFFF):08X})")
    parts.append(f"Executable: {executable}")

    if returncode in (-1073741515, 3221225781):
        parts.append(
            "Windows reports STATUS_DLL_NOT_FOUND (0xC0000135).  This ccx "
            "binary needs a runtime DLL that is not next to it on PATH.  The "
            "'ccx_dynamic.exe' build requires Intel MKL's mkl_rt.2.dll; use "
            "'ccx_static.exe' instead (already preferred by install_solvers.py "
            "as of CalculiX 2.23).  Re-run:  python scripts/install_solvers.py "
            "--only ccx  to refresh the path."
        )

    for ext in (".sta", ".cvg", ".log", ".dat"):
        path = work_dir / f"{job_name}{ext}"
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        content = content.strip()
        if not content:
            continue
        parts.append(f"--- {path.name} (last 2000 chars) ---\n{tail(content, 2000)}")
    return "\n".join(parts)


def _build_sets_and_step(
    mesh: Any,
    constraints: List[dict],
    loads: List[dict],
    warnings: List[str],
) -> Tuple[List[str], List[str]]:
    """Build CalculiX node sets, surface sets, and step records."""
    model_lines: List[str] = []
    step_lines: List[str] = ["*STEP", "*STATIC"]

    boundary_lines: List[str] = []
    for idx, constraint in enumerate(constraints, start=1):
        geoms = dict_geometries(constraint)
        condition = str(constraint.get("condition", "") or "").strip()
        if geoms:
            node_ids = nodes_matching_geometries(mesh, geoms) + 1
        elif condition:
            node_ids = nodes_matching_condition(
                mesh, condition, warnings, label=f"Constraint {idx}"
            ) + 1
        else:
            warnings.append(f"Constraint {idx} has no selected face geometry or condition.")
            continue
        if len(node_ids) == 0:
            warnings.append(f"Constraint {idx} did not match any mesh nodes.")
            continue

        set_name = f"BC_{idx}"
        model_lines.append(f"*NSET, NSET={set_name}")
        model_lines.extend(id_lines(node_ids))

        fixed_dofs = constraint.get("fixed_dofs", [0, 1, 2])
        disp = constraint.get("displacement", None)
        for dof_idx in fixed_dofs:
            value = 0.0 if disp is None else float(disp[int(dof_idx)])
            ccx_dof = int(dof_idx) + 1
            boundary_lines.append(f"{set_name}, {ccx_dof}, {ccx_dof}, {value:.12g}")

    if boundary_lines:
        step_lines.append("*BOUNDARY")
        step_lines.extend(boundary_lines)
    else:
        warnings.append("No boundary constraints were exported to the CalculiX deck.")

    cload_lines: List[str] = []
    dload_lines: List[str] = []
    for idx, load in enumerate(loads, start=1):
        ltype = load.get("type", "force")
        if ltype == "force":
            geoms = dict_geometries(load)
            condition = str(load.get("condition", "") or "").strip()
            if geoms:
                node_ids = nodes_matching_geometries(mesh, geoms) + 1
            elif condition:
                node_ids = nodes_matching_condition(
                    mesh, condition, warnings, label=f"Force load {idx}"
                ) + 1
            else:
                warnings.append(f"Force load {idx} has no selected face geometry or condition.")
                continue
            if len(node_ids) == 0:
                warnings.append(f"Force load {idx} did not match any mesh nodes.")
                continue
            force = load_vector(load)
            nodal_force = force / max(len(node_ids), 1)
            for node_id in node_ids:
                for dof_idx, value in enumerate(nodal_force, start=1):
                    if abs(float(value)) > 1e-16:
                        cload_lines.append(f"{int(node_id)}, {dof_idx}, {float(value):.12g}")
        elif ltype == "gravity":
            direction = load.get("direction", "-Y")
            dir_map = {
                "-X": (-1.0, 0.0, 0.0),
                "+X": (1.0, 0.0, 0.0),
                "-Y": (0.0, -1.0, 0.0),
                "+Y": (0.0, 1.0, 0.0),
                "-Z": (0.0, 0.0, -1.0),
                "+Z": (0.0, 0.0, 1.0),
            }
            dx, dy, dz = dir_map.get(direction, (0.0, -1.0, 0.0))
            dload_lines.append(
                f"EALL, GRAV, {float(load.get('accel', 9810.0)):.12g}, {dx:.1f}, {dy:.1f}, {dz:.1f}"
            )
        elif ltype == "pressure":
            geoms = dict_geometries(load)
            if not geoms:
                warnings.append(
                    f"Pressure load {idx} has no selected face geometry; skipped."
                )
                continue
            faces = tet_face_sets_for_geometries(mesh, geoms)
            if not faces:
                warnings.append(
                    f"Pressure load {idx}: no external tet faces matched the selected geometry. "
                    "Check that the selected face is a boundary of the mesh."
                )
                continue
            set_name = f"PRESS_{idx}"
            model_lines.append(f"*SURFACE, NAME={set_name}, TYPE=ELEMENT")
            for elem_id, face_id in faces:
                model_lines.append(f"{int(elem_id)}, S{int(face_id)}")
            pressure = float(load.get("pressure", load.get("magnitude", 0.0)))
            dload_lines.append(f"{set_name}, P, {pressure:.12g}")
        else:
            warnings.append(f"Unsupported CalculiX load type: {ltype}")

    if cload_lines:
        step_lines.append("*CLOAD")
        step_lines.extend(cload_lines)
    if dload_lines:
        step_lines.append("*DLOAD")
        step_lines.extend(dload_lines)
    if not cload_lines and not dload_lines:
        warnings.append("No external loads were exported to the CalculiX deck.")

    step_lines.extend(
        [
            "*NODE FILE",
            "U, RF",
            "*EL FILE",
            "S, E, ENER",
            "*END STEP",
        ]
    )
    return model_lines, step_lines


def _material_block(material: dict) -> List[str]:
    """Write the *MATERIAL block including *PLASTIC when yield strength is set.

    A non-zero ``yield_strength`` in the material dict produces a bilinear
    isotropic-hardening *PLASTIC table:

        σ_y at εp = 0      → ``yield_strength``
        σ_y at εp = ``ε*`` → ``yield_strength + tangent_modulus · ε*``

    where ``ε* = 0.10`` is a representative plastic-strain anchor.  When
    ``yield_strength <= 0`` the material is pure linear-elastic and no
    *PLASTIC card is emitted — CalculiX then runs a linear elastic problem.
    """
    e   = float(material.get("E", 210000.0))
    nu  = float(material.get("nu", material.get("poissons_ratio", 0.3)))
    rho = float(material.get("rho", material.get("density", 7.85e-9)))
    sigma_y = float(material.get("yield_strength", 0.0) or 0.0)
    et      = float(material.get("tangent_modulus", 0.0) or 0.0)

    lines: List[str] = [
        "*MATERIAL, NAME=MAT1",
        "*ELASTIC",
        f"{e:.12g}, {nu:.12g}",
        "*DENSITY",
        f"{rho:.12g}",
    ]
    if sigma_y > 0.0:
        eps_anchor = 0.10
        sigma_anchor = sigma_y + et * eps_anchor
        lines.extend(
            [
                "*PLASTIC, HARDENING=ISOTROPIC",
                f"{sigma_y:.12g}, 0.0",
                f"{sigma_anchor:.12g}, {eps_anchor:.6g}",
            ]
        )
    lines.append("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1")
    lines.append("")
    return lines


def _step_header(analysis_type: str) -> List[str]:
    """Return the *STEP / *STATIC header lines for the requested analysis.

    ``analysis_type``:
      - ``'Linear'``                 → ``*STEP`` + bare ``*STATIC``
      - ``'Nonlinear (Geometric)'``  → ``*STEP, NLGEOM`` + incremented ``*STATIC``
      - ``'Nonlinear (Plastic)'``    → same as Geometric (CalculiX auto-enables
                                       NLGEOM when *PLASTIC is present, but
                                       writing it explicitly keeps intent
                                       readable in the deck)
    """
    if analysis_type == "Linear":
        return ["*STEP", "*STATIC"]
    # Incremented static — 10 increments by default, with adaptive sub-stepping
    # on convergence trouble.  Init/total/min/max increment lengths follow the
    # *STATIC card spec: initial_inc, total_step, min_inc, max_inc.
    return [
        "*STEP, NLGEOM, INC=200",
        "*STATIC",
        "0.1, 1.0, 1e-5, 1.0",
    ]


def _build_input_deck(
    mesh: Any,
    material: dict,
    constraints: List[dict],
    loads: List[dict],
    warnings: List[str],
    analysis_type: str = "Linear",
) -> str:
    """Create a CalculiX/Abaqus-style input deck.

    ``analysis_type`` selects linear vs nonlinear (NLGEOM) static.  Plasticity
    is auto-enabled whenever ``material['yield_strength'] > 0``.
    """
    points, tets = mesh_to_tet4(mesh, warnings)
    step_header = _step_header(analysis_type)
    # Note: _build_sets_and_step prepends a hard-coded ``*STEP``/``*STATIC``;
    # we override it below by replacing the leading two entries with the
    # analysis-specific header.
    model_lines, step_lines = _build_sets_and_step(mesh, constraints, loads, warnings)
    if step_lines[:2] == ["*STEP", "*STATIC"]:
        step_lines = step_header + step_lines[2:]

    lines: List[str] = [
        "*HEADING",
        f"PyLCSS CalculiX deck ({analysis_type})",
        "*NODE",
    ]
    for idx, xyz in enumerate(points, start=1):
        lines.append(f"{idx}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}")

    lines.append("*ELEMENT, TYPE=C3D4, ELSET=EALL")
    for idx, conn in enumerate(tets, start=1):
        node_ids = [int(v) + 1 for v in conn]
        lines.append(f"{idx}, {node_ids[0]}, {node_ids[1]}, {node_ids[2]}, {node_ids[3]}")

    lines.extend(_material_block(material))
    lines.extend(model_lines)
    lines.extend(step_lines)
    return "\n".join(lines) + "\n"


def _ingest_frd_into_result(
    mesh: Any,
    frd_path: Path,
    visualization_mode: str,
    warnings: List[str],
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
    n_frd = frd_node_ids.size

    if n_frd != n_points:
        warnings.append(
            f"CalculiX FRD reports {n_frd} nodes; PyLCSS source mesh has "
            f"{n_points}.  Scattering by node id; any missing ids will render "
            "with zero displacement / zero stress."
        )

    # Scatter by id (CCX writes 1-based ids; mesh array is 0-based).
    disp_xyz = np.zeros((n_points, 3), dtype=float)
    vm = np.zeros(n_points, dtype=float)
    stress_tensor = np.zeros((n_points, 6), dtype=float)
    ener_nodal = np.zeros(n_points, dtype=float)
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

    # Viewer expects flat (3*N,) reshaping to (3, N) with order='F'.
    flat_disp = np.zeros(3 * n_points, dtype=float)
    flat_disp[0::3] = disp_xyz[:, 0]
    flat_disp[1::3] = disp_xyz[:, 1]
    flat_disp[2::3] = disp_xyz[:, 2]

    max_disp = float(np.max(np.abs(disp_xyz))) if disp_xyz.size else 0.0
    peak_vm = float(np.max(vm)) if vm.size else 0.0

    # Tet element volumes (used to integrate the nodal ENER density into the
    # total elastic strain energy and to project ENER back per-element for SIMP).
    tets = np.asarray(mesh.t).T[:, :4].astype(int)
    coords = np.asarray(mesh.p).T
    v0 = coords[tets[:, 0]]
    edges = np.stack(
        [coords[tets[:, 1]] - v0, coords[tets[:, 2]] - v0, coords[tets[:, 3]] - v0],
        axis=-1,
    )
    elem_vol = np.abs(np.linalg.det(edges)) / 6.0

    # Per-element ENER = mean of the 4 nodal density values; matches CCX's
    # nodal-averaging convention so totals integrate consistently.
    elem_ener = ener_nodal[tets].mean(axis=1) if ener_nodal.size else np.zeros(tets.shape[0])

    # Total elastic strain energy ≈ Σ_e (ENER_e × V_e); compliance = 2 × SE for
    # linear elasticity (C = u·f = u·K·u = 2·SE).
    total_strain_energy = float(np.sum(elem_ener * elem_vol))
    compliance = 2.0 * total_strain_energy

    print(
        f"FEA Solver (external): FRD ingest "
        f"frd_nodes={n_frd}, mesh_nodes={n_points}, "
        f"max|u|={max_disp:.4e}, peak VM={peak_vm:.4e}, "
        f"compliance={compliance:.4e}"
    )

    # Auto deformation scale so deformed shape is visible without misleading.
    bbox = np.ptp(np.asarray(mesh.p), axis=1)
    char_len = max(float(np.max(bbox)), 1e-9)
    auto_scale = (0.05 * char_len / max_disp) if max_disp > 1e-9 else 1.0
    deformation_scale = float(np.clip(auto_scale, 1.0, 200.0))

    total_volume = float(np.sum(elem_vol))

    return {
        "type": "fea",
        "backend": "CalculiX",
        "mesh": mesh,
        "displacement": flat_disp,
        "stress": vm,
        "stress_tensor": stress_tensor,
        "ener_nodal": ener_nodal,
        "element_ener": elem_ener,
        "element_volumes": elem_vol,
        "strain_energy": total_strain_energy,
        "compliance": compliance,
        "volume": total_volume,
        "peak_displacement": max_disp,
        "max_stress_gauss": peak_vm,
        "deformation_scale": deformation_scale,
        "visualization_mode": visualization_mode,
        "frd_file": str(frd_path),
        "frd_steps": parsed.get("steps", []),
    }


def _build_topopt_input_deck(
    mesh: Any,
    material: dict,
    constraints: List[dict],
    loads: List[dict],
    densities: np.ndarray,
    p_penal: float,
    rho_min: float,
    n_bins: int,
    warnings: List[str],
) -> str:
    """Build a CalculiX deck where elements are grouped by SIMP-binned modulus.

    Element modulus follows the modified-SIMP interpolation
        E_e = (rho_min + (1 - rho_min) * rho_e**p) * E_0
    and is quantised into ``n_bins`` discrete materials so the deck stays
    tractable on meshes with > 10⁵ elements.
    """
    points, tets = mesh_to_tet4(mesh, warnings)
    model_lines, step_lines = _build_sets_and_step(mesh, constraints, loads, warnings)

    n_elem = tets.shape[0]
    if densities.shape[0] != n_elem:
        raise SolverBackendError(
            f"densities has {densities.shape[0]} entries but mesh has {n_elem} tets."
        )

    rho_clipped = np.clip(densities.astype(float), float(rho_min), 1.0)
    e_factor = float(rho_min) + (1.0 - float(rho_min)) * (rho_clipped ** float(p_penal))

    # Logarithmic binning over the modulus *factor* so the dense-end resolution
    # (where sensitivities matter most for compliance) is preserved.
    e_lo = float(np.min(e_factor))
    e_hi = float(np.max(e_factor))
    if e_hi - e_lo < 1e-9:
        bin_ids = np.zeros(n_elem, dtype=int)
        bin_values = np.array([e_hi])
    else:
        edges = np.linspace(np.log(max(e_lo, 1e-12)), np.log(e_hi), int(n_bins) + 1)
        bin_ids = np.clip(
            np.searchsorted(edges[1:-1], np.log(e_factor)),
            0,
            int(n_bins) - 1,
        )
        bin_centers_log = 0.5 * (edges[:-1] + edges[1:])
        bin_values = np.exp(bin_centers_log)

    e_base = float(material.get("E", 210000.0))
    nu = float(material.get("nu", material.get("poissons_ratio", 0.3)))
    rho_dens = float(material.get("rho", material.get("density", 7.85e-9)))

    lines: List[str] = [
        "*HEADING",
        "PyLCSS CalculiX SIMP topology-optimisation iteration",
        "*NODE",
    ]
    for idx, xyz in enumerate(points, start=1):
        lines.append(f"{idx}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}")

    lines.append("*ELEMENT, TYPE=C3D4, ELSET=EALL")
    for idx, conn in enumerate(tets, start=1):
        node_ids = [int(v) + 1 for v in conn]
        lines.append(f"{idx}, {node_ids[0]}, {node_ids[1]}, {node_ids[2]}, {node_ids[3]}")

    # Per-bin element sets (1-based CCX element ids).
    used_bins = np.unique(bin_ids)
    for b in used_bins:
        elset = f"E_BIN_{b}"
        elem_ids = np.where(bin_ids == b)[0] + 1
        lines.append(f"*ELSET, ELSET={elset}")
        lines.extend(id_lines(elem_ids))

    # Per-bin material + solid section.
    for b in used_bins:
        e_b = float(bin_values[b]) * e_base
        mat_name = f"MAT_{b}"
        lines.extend(
            [
                f"*MATERIAL, NAME={mat_name}",
                "*ELASTIC",
                f"{e_b:.12g}, {nu:.12g}",
                "*DENSITY",
                f"{rho_dens:.12g}",
                f"*SOLID SECTION, ELSET=E_BIN_{b}, MATERIAL={mat_name}",
                "",
            ]
        )
    lines.extend(model_lines)
    lines.extend(step_lines)
    return "\n".join(lines) + "\n"


def run_calculix_topopt_iteration(
    mesh: Any,
    material: dict,
    constraints: List[dict],
    loads: List[dict],
    densities: np.ndarray,
    p_penal: float,
    rho_min: float,
    config: ExternalRunConfig,
    n_bins: int = 32,
) -> dict:
    """Run one SIMP topology-optimisation iteration through CalculiX.

    Returns the standard FEA result dict plus ``element_ener`` and
    ``element_volumes`` for SIMP sensitivity computation.
    """
    warnings: List[str] = []
    work_dir = make_work_dir("pylcss_calculix_topopt_", config.work_dir)
    job_name = config.job_name or "pylcss_calculix_topopt"
    inp_path = work_dir / f"{job_name}.inp"
    inp_path.write_text(
        _build_topopt_input_deck(
            mesh, material, constraints, loads,
            np.asarray(densities, dtype=float),
            p_penal, rho_min, n_bins, warnings,
        ),
        encoding="utf-8",
    )

    executable = resolve_executable(
        config.executable,
        env_vars=("PYLCSS_CALCULIX_CCX", "CALCULIX_CCX"),
        candidates=(
            "ccx_static", "ccx_static.exe",
            "ccx", "ccx.exe",
            "ccx_dynamic", "ccx_dynamic.exe",
        ),
    )

    status = "deck_written"
    solver_log = ""
    result_payload: dict = {}
    if config.run_solver:
        if executable is None:
            raise SolverBackendError(
                "CalculiX executable not found. Set the node path, add ccx to PATH, "
                "define PYLCSS_CALCULIX_CCX, or run scripts/install_solvers.py."
            )
        exe_dir = str(Path(executable).resolve().parent)
        proc = run_process(
            [executable, job_name],
            cwd=work_dir,
            timeout_s=config.timeout_s,
            extra_path_dirs=(exe_dir,),
        )
        solver_log = tail(proc.stdout or "")
        if proc.returncode != 0:
            aux = _collect_calculix_failure_context(work_dir, job_name, proc.returncode, executable)
            raise SolverBackendError(
                "CalculiX (topopt iteration) failed. Last solver output:\n"
                + (solver_log or "(stdout was empty)\n")
                + "\n"
                + aux
            )
        status = "completed"
        frd_path = work_dir / f"{job_name}.frd"
        if not frd_path.is_file():
            raise SolverBackendError(
                f"CalculiX (topopt iteration) produced no {frd_path.name}; "
                "check the .dat / .sta logs in the work directory."
            )
        result_payload = _ingest_frd_into_result(mesh, frd_path, "Density", warnings)
        rho_mat = float(material.get("rho", material.get("density", 0.0)))
        if rho_mat > 0.0 and "element_volumes" in result_payload:
            phys = np.asarray(densities, dtype=float)
            elem_vol_arr = np.asarray(result_payload["element_volumes"], dtype=float)
            result_payload["mass"] = float(np.sum(phys * elem_vol_arr) * rho_mat)

    output = {
        "type": result_payload.get("type", "external_solver"),
        "backend": "CalculiX (SIMP)",
        "external_status": status,
        "mesh": mesh,
        "input_file": str(inp_path),
        "work_dir": str(work_dir),
        "solver_executable": executable,
        "solver_log": solver_log,
        "warnings": warnings,
    }
    output.update(result_payload)
    return output


def run_calculix_static(
    mesh: Any,
    material: dict,
    constraints: List[dict],
    loads: List[dict],
    config: ExternalRunConfig,
    visualization_mode: str = "Von Mises Stress",
    analysis_type: str = "Linear",
) -> dict:
    """Write and optionally run a CalculiX static analysis deck.

    ``analysis_type`` is one of ``'Linear'``, ``'Nonlinear (Geometric)'``,
    or ``'Nonlinear (Plastic)'``.  Plasticity is also auto-enabled whenever
    the connected material dict carries ``yield_strength > 0``.
    """
    warnings: List[str] = []
    work_dir = make_work_dir("pylcss_calculix_", config.work_dir)
    job_name = config.job_name or "pylcss_calculix"
    inp_path = work_dir / f"{job_name}.inp"
    inp_path.write_text(
        _build_input_deck(
            mesh, material, constraints, loads, warnings,
            analysis_type=analysis_type,
        ),
        encoding="utf-8",
    )

    executable = resolve_executable(
        config.executable,
        env_vars=("PYLCSS_CALCULIX_CCX", "CALCULIX_CCX"),
        # Prefer the self-contained static build on every platform.  The
        # ``ccx_dynamic`` variant exists for users who have Intel MKL installed
        # globally; we only fall through to it last.
        candidates=(
            "ccx_static",
            "ccx_static.exe",
            "ccx",
            "ccx.exe",
            "ccx_dynamic",
            "ccx_dynamic.exe",
        ),
    )

    status = "deck_written"
    solver_log = ""
    result_payload: dict = {}
    if config.run_solver:
        if executable is None:
            warnings.append(
                "CalculiX executable not found. Set the node path, add ccx to PATH, "
                "define PYLCSS_CALCULIX_CCX, or run scripts/install_solvers.py."
            )
        else:
            exe_dir = str(Path(executable).resolve().parent)
            proc = run_process(
                [executable, job_name],
                cwd=work_dir,
                timeout_s=config.timeout_s,
                extra_path_dirs=(exe_dir,),
            )
            solver_log = tail(proc.stdout or "")
            if proc.returncode != 0:
                # CCX writes its own diagnostics to .sta / .dat even when stdout
                # is empty (which it always is when the .exe fails to load due
                # to a missing DLL).  Capture those for the user.
                aux = _collect_calculix_failure_context(work_dir, job_name, proc.returncode, executable)
                raise SolverBackendError(
                    "CalculiX failed. Last solver output:\n"
                    + (solver_log or "(stdout was empty)\n")
                    + "\n"
                    + aux
                )
            status = "completed"
            frd_path = work_dir / f"{job_name}.frd"
            if not frd_path.is_file():
                warnings.append(
                    f"CalculiX run completed but {frd_path.name} was not produced. "
                    "Check the .dat / .sta logs in the work directory."
                )
            else:
                try:
                    result_payload = _ingest_frd_into_result(
                        mesh, frd_path, visualization_mode, warnings
                    )
                    rho = float(material.get("rho", material.get("density", 0.0)))
                    if rho > 0.0 and "volume" in result_payload:
                        result_payload["mass"] = float(result_payload["volume"] * rho)
                except Exception as exc:  # parser problems should not kill the node
                    warnings.append(f"FRD ingest failed: {exc}")

    output = {
        "type": result_payload.get("type", "external_solver"),
        "backend": "CalculiX",
        "external_status": status,
        "mesh": mesh,
        "visualization_mode": visualization_mode,
        "input_file": str(inp_path),
        "work_dir": str(work_dir),
        "solver_executable": executable,
        "solver_log": solver_log,
        "warnings": warnings,
        "message": (
            "CalculiX deck generated. Enable external execution and configure ccx "
            "to run the solve from PyLCSS."
            if status == "deck_written"
            else (
                "CalculiX run complete; results imported."
                if result_payload
                else "CalculiX run finished but no FRD results could be loaded."
            )
        ),
    }
    output.update(result_payload)
    return output
