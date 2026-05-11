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
        if not geoms:
            warnings.append(
                f"Constraint {idx} has no selected face geometry; condition-string "
                "constraints are not exported to CalculiX yet."
            )
            continue
        node_ids = nodes_matching_geometries(mesh, geoms) + 1
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
            if not geoms:
                warnings.append(
                    f"Force load {idx} has no selected face geometry; condition-string "
                    "loads are not exported to CalculiX yet."
                )
                continue
            node_ids = nodes_matching_geometries(mesh, geoms) + 1
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
            "S, E",
            "*END STEP",
        ]
    )
    return model_lines, step_lines


def _build_input_deck(
    mesh: Any,
    material: dict,
    constraints: List[dict],
    loads: List[dict],
    warnings: List[str],
) -> str:
    """Create a CalculiX/Abaqus-style input deck for linear static solids."""
    points, tets = mesh_to_tet4(mesh, warnings)
    model_lines, step_lines = _build_sets_and_step(mesh, constraints, loads, warnings)

    e = float(material.get("E", 210000.0))
    nu = float(material.get("nu", material.get("poissons_ratio", 0.3)))
    rho = float(material.get("rho", material.get("density", 7.85e-9)))

    lines: List[str] = [
        "*HEADING",
        "PyLCSS CalculiX static structural deck",
        "*NODE",
    ]
    for idx, xyz in enumerate(points, start=1):
        lines.append(f"{idx}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}")

    lines.append("*ELEMENT, TYPE=C3D4, ELSET=EALL")
    for idx, conn in enumerate(tets, start=1):
        node_ids = [int(v) + 1 for v in conn]
        lines.append(f"{idx}, {node_ids[0]}, {node_ids[1]}, {node_ids[2]}, {node_ids[3]}")

    lines.extend(
        [
            "*MATERIAL, NAME=MAT1",
            "*ELASTIC",
            f"{e:.12g}, {nu:.12g}",
            "*DENSITY",
            f"{rho:.12g}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1",
            "",
        ]
    )
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
    if n_frd > 0:
        mesh_idx = frd_node_ids - 1
        valid = (mesh_idx >= 0) & (mesh_idx < n_points)
        if not valid.all():
            n_invalid = int((~valid).sum())
            warnings.append(
                f"{n_invalid} FRD node id(s) fall outside the source-mesh range; ignored."
            )
        idx_in = mesh_idx[valid]
        if disp_xyz_frd.shape[0] >= valid.size:
            disp_xyz[idx_in] = disp_xyz_frd[np.where(valid)[0]]
        if vm_frd.size >= valid.size:
            vm[idx_in] = vm_frd[np.where(valid)[0]]

    # Viewer expects flat (3*N,) reshaping to (3, N) with order='F'.
    flat_disp = np.zeros(3 * n_points, dtype=float)
    flat_disp[0::3] = disp_xyz[:, 0]
    flat_disp[1::3] = disp_xyz[:, 1]
    flat_disp[2::3] = disp_xyz[:, 2]

    max_disp = float(np.max(np.abs(disp_xyz))) if disp_xyz.size else 0.0
    peak_vm = float(np.max(vm)) if vm.size else 0.0
    print(
        f"FEA Solver (external): FRD ingest "
        f"frd_nodes={n_frd}, mesh_nodes={n_points}, "
        f"max|u|={max_disp:.4e}, peak VM={peak_vm:.4e}"
    )

    # Auto deformation scale so deformed shape is visible without misleading.
    bbox = np.ptp(np.asarray(mesh.p), axis=1)
    char_len = max(float(np.max(bbox)), 1e-9)
    auto_scale = (0.05 * char_len / max_disp) if max_disp > 1e-9 else 1.0
    deformation_scale = float(np.clip(auto_scale, 1.0, 200.0))

    return {
        "type": "fea",
        "backend": "CalculiX",
        "mesh": mesh,
        "displacement": flat_disp,
        "stress": vm,
        "max_stress_gauss": peak_vm,
        "deformation_scale": deformation_scale,
        "visualization_mode": visualization_mode,
        "frd_file": str(frd_path),
        "frd_steps": parsed.get("steps", []),
    }


def run_calculix_static(
    mesh: Any,
    material: dict,
    constraints: List[dict],
    loads: List[dict],
    config: ExternalRunConfig,
    visualization_mode: str = "Von Mises Stress",
) -> dict:
    """Write and optionally run a CalculiX linear static analysis deck."""
    warnings: List[str] = []
    work_dir = make_work_dir("pylcss_calculix_", config.work_dir)
    job_name = config.job_name or "pylcss_calculix"
    inp_path = work_dir / f"{job_name}.inp"
    inp_path.write_text(
        _build_input_deck(mesh, material, constraints, loads, warnings),
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
