# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""CalculiX modal analysis backend (natural frequencies and mode shapes)."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from pylcss.cae.solver_backends.base import (
    ExternalRunConfig,
    SolverBackendError,
)
from pylcss.cae.solver_backends.calculix import (
    _collect_calculix_failure_context,
    resolve_calculix_executable,
)
from pylcss.cae.solver_backends.calculix_results import _resolve_deformation_scale
from pylcss.cae.solver_backends.execution import make_work_dir, run_process, tail
from pylcss.cae.solver_backends.frd_reader import _parse_data_record
from pylcss.cae.solver_backends.mesh import (
    id_lines,
    mesh_to_tet4,
    mesh_to_tet10,
    tet10_connectivity,
)
from pylcss.cae.solver_backends.selection import (
    dict_geometries,
    nodes_matching_condition,
    nodes_matching_geometries,
)
from pylcss.cae.solver_backends.validation import (
    integer,
    positive_float,
    record_list,
    validate_isotropic_material,
)

logger = logging.getLogger(__name__)


def _build_modal_deck(
    mesh: Any,
    material: dict,
    constraints: list[dict],
    num_modes: int,
    warnings: list[str],
) -> str:
    """Generate a CalculiX input deck for natural frequency extraction."""
    validate_isotropic_material(material)
    rho = float(material.get("rho", material.get("density", 0.0)) or 0.0)
    if not math.isfinite(rho) or rho <= 0.0:
        raise SolverBackendError(
            "Modal analysis requires a non-zero material density (Material.density > 0)."
        )

    lines: list[str] = ["*HEADING", "PyLCSS CalculiX Modal Analysis"]

    # Mesh nodes and elements
    is_quadratic = tet10_connectivity(mesh) is not None
    if is_quadratic:
        pts, cells = mesh_to_tet10(mesh, warnings)
        elem_type = "C3D10"
    else:
        pts, cells = mesh_to_tet4(mesh, warnings)
        elem_type = "C3D4"

    lines.append("*NODE, NSET=NALL")
    for idx, xyz in enumerate(pts, start=1):
        lines.append(f"{idx}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}")

    lines.append(f"*ELEMENT, TYPE={elem_type}, ELSET=EALL")
    for idx, conn in enumerate(cells, start=1):
        node_strs = ", ".join(str(int(node_idx) + 1) for node_idx in conn)
        lines.append(f"{idx}, {node_strs}")

    # Material definition
    # MaterialNode's public structural payload uses the compact CalculiX keys
    # E/nu/rho.  Also accept the descriptive aliases used by external callers.
    youngs_modulus = positive_float(
        material.get("E", material.get("youngs_modulus", 210000.0)),
        label="Young's modulus",
    )
    poissons_ratio = float(material.get("nu", material.get("poissons_ratio", 0.3)))
    lines.extend(
        [
            "*MATERIAL, NAME=MAT1",
            "*ELASTIC",
            f"{youngs_modulus:.12g}, {poissons_ratio:.12g}",
            "*DENSITY",
            f"{rho:.12g}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1",
        ]
    )

    # Boundary conditions
    boundary_lines: list[str] = []
    for idx, constraint in enumerate(constraints, start=1):
        geoms = dict_geometries(constraint)
        condition = str(constraint.get("condition", "") or "").strip()
        if geoms:
            node_ids = nodes_matching_geometries(mesh, geoms) + 1
        elif condition:
            node_ids = (
                nodes_matching_condition(
                    mesh, condition, warnings, label=f"Constraint {idx}"
                )
                + 1
            )
        else:
            raise SolverBackendError(
                f"Constraint {idx} has no target geometry or condition."
            )
        if len(node_ids) == 0:
            raise SolverBackendError(f"Constraint {idx} matches no mesh nodes.")

        set_name = f"BC_{idx}"
        lines.append(f"*NSET, NSET={set_name}")
        lines.extend(id_lines(node_ids))

        fixed_dofs = [
            integer(value, label=f"Constraint {idx} DOF")
            for value in constraint.get("fixed_dofs", [0, 1, 2])
        ]
        for dof_idx in fixed_dofs:
            if dof_idx not in (0, 1, 2):
                raise SolverBackendError(f"Constraint {idx} DOF must be 0, 1, or 2.")
            ccx_dof = dof_idx + 1
            boundary_lines.append(f"{set_name}, {ccx_dof}, {ccx_dof}, 0.0")

    # Analysis step
    lines.extend(["*STEP", "*FREQUENCY", str(max(1, int(num_modes)))])
    if boundary_lines:
        lines.append("*BOUNDARY")
        lines.extend(boundary_lines)
    else:
        warnings.append(
            "No boundary constraints specified: running free-free modal analysis "
            "(first 6 modes will be rigid-body modes with ~0 Hz)."
        )

    lines.extend(["*NODE FILE", "U", "*END STEP"])
    return "\n".join(lines) + "\n"


def _parse_modal_dat(dat_path: Path) -> list[dict[str, Any]]:
    """Parse natural frequencies from CalculiX .dat output file."""
    if not dat_path.is_file():
        return []

    content = dat_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    modes: list[dict[str, Any]] = []

    in_eigen_table = False
    for line in lines:
        stripped = line.strip()
        normalized = stripped.replace(" ", "").upper()
        if "EIGENVALUEOUTPUT" in normalized:
            in_eigen_table = True
            continue
        if in_eigen_table:
            if not stripped:
                if modes:
                    # End of eigenvalue table
                    break
                continue
            if (
                "PARTICIPATION" in normalized
                or "EFFECTIVEMODALMASS" in normalized
                or "STEP" in normalized
            ):
                break
            tokens = stripped.split()
            # Expect: MODE_NO, EIGENVALUE, FREQ_RAD_TIME, FREQ_CYCLES_TIME
            if len(tokens) >= 4 and tokens[0].isdigit():
                try:
                    mode_no = int(tokens[0])
                    if mode_no != len(modes) + 1:
                        # Reached another section with mode numbers
                        break
                    eigenvalue = float(tokens[1].replace("D", "E").replace("d", "e"))
                    freq_rad = float(tokens[2].replace("D", "E").replace("d", "e"))
                    freq_hz = float(tokens[3].replace("D", "E").replace("d", "e"))
                    period_s = (1.0 / freq_hz) if freq_hz > 1e-9 else float("inf")
                    modes.append(
                        {
                            "mode": mode_no,
                            "eigenvalue": eigenvalue,
                            "frequency_rad_s": freq_rad,
                            "frequency_hz": freq_hz,
                            "period_s": period_s,
                        }
                    )
                except ValueError:
                    pass

    return modes


def _parse_modal_frd(frd_path: Path, n_points: int) -> dict[int, np.ndarray]:
    """Parse mode shape eigenvectors from CalculiX .frd output file.

    Returns dict mapping 1-based mode index -> (n_points, 3) displacement array.
    """
    if not frd_path.is_file():
        return {}

    content = frd_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    mode_shapes: dict[int, np.ndarray] = {}
    current_mode: int | None = None
    reading_disp = False
    disp_dict: dict[int, list[float]] = {}

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if stripped.startswith("1PMODE"):
            tokens = stripped.split()
            if len(tokens) >= 2:
                try:
                    current_mode = int(tokens[1])
                except ValueError:
                    current_mode = len(mode_shapes) + 1
            idx += 1
            continue

        if stripped.startswith("1PSTEP") and current_mode is None:
            tokens = stripped.split()
            if len(tokens) >= 2:
                try:
                    current_mode = int(tokens[1])
                except ValueError:
                    pass

        if stripped.startswith("-4") and "DISP" in stripped.upper():
            reading_disp = True
            disp_dict = {}
            idx += 1
            continue

        if reading_disp:
            if stripped.startswith("-3"):
                reading_disp = False
                mode_idx = (
                    current_mode if current_mode is not None else (len(mode_shapes) + 1)
                )
                # Scatter by 1-based node id into (n_points, 3) array
                shape_arr = np.zeros((n_points, 3), dtype=float)
                for nid, uvw in disp_dict.items():
                    if 1 <= nid <= n_points:
                        shape_arr[nid - 1] = uvw[:3]
                mode_shapes[mode_idx] = shape_arr
                current_mode = None
                idx += 1
                continue

            if stripped.startswith("-1"):
                nid, vals = _parse_data_record(line)
                if nid > 0 and len(vals) >= 3:
                    disp_dict[nid] = vals[:3]
        idx += 1

    return mode_shapes


def run_calculix_modal(
    mesh: Any,
    material: dict,
    constraints: list[dict] | None = None,
    num_modes: int = 6,
    active_mode: int = 1,
    config: ExternalRunConfig | None = None,
    deformation_scale: Any = "Auto",
) -> dict[str, Any]:
    """Execute CalculiX modal analysis (*FREQUENCY) and return frequencies and mode shapes."""
    if config is None:
        config = ExternalRunConfig(run_solver=True)
    warnings: list[str] = []
    constraints_list = record_list(constraints or [], label="Constraints")

    job_name = config.validated_job_name(default="pylcss_modal")
    timeout_s = config.validated_timeout() if config.run_solver else config.timeout_s
    work_dir = make_work_dir("pylcss_modal_", config.work_dir)
    inp_path = work_dir / f"{job_name}.inp"

    deck_str = _build_modal_deck(mesh, material, constraints_list, num_modes, warnings)
    try:
        inp_path.write_text(deck_str, encoding="utf-8")
    except OSError as exc:
        raise SolverBackendError(
            f"Could not write CalculiX modal deck {inp_path}: {exc}"
        ) from exc

    executable = resolve_calculix_executable(config.executable)
    status = "deck_written"
    modes: list[dict[str, Any]] = []
    mode_shapes: dict[int, np.ndarray] = {}
    mode_displacements_flat: dict[int, np.ndarray] = {}
    solver_elapsed_s: float | None = None

    if config.run_solver:
        if executable is None:
            raise SolverBackendError(
                "CalculiX executable not found. Set the node path, add ccx to PATH, "
                "or run scripts/install_solvers.py."
            )
        exe_dir = str(Path(executable).resolve().parent)
        started = time.perf_counter()
        proc = run_process(
            [executable, job_name],
            cwd=work_dir,
            timeout_s=timeout_s,
            extra_path_dirs=(exe_dir,),
            cancel_callback=config.cancel_callback,
        )
        solver_elapsed_s = time.perf_counter() - started
        solver_log = tail(proc.stdout or "")
        if proc.returncode != 0:
            aux = _collect_calculix_failure_context(
                work_dir, job_name, proc.returncode, executable
            )
            raise SolverBackendError(
                "CalculiX modal solve failed. Last solver output:\n"
                + (solver_log or "(stdout was empty)\n")
                + "\n"
                + aux
            )
        status = "completed"

        dat_path = work_dir / f"{job_name}.dat"
        frd_path = work_dir / f"{job_name}.frd"

        modes = _parse_modal_dat(dat_path)
        n_points = int(np.asarray(mesh.p).shape[1])
        mode_shapes = _parse_modal_frd(frd_path, n_points)
        if not modes:
            raise SolverBackendError(
                "CalculiX returned no natural frequencies. Inspect the solver files in "
                + str(work_dir)
            )
        missing = [m["mode"] for m in modes if m["mode"] not in mode_shapes]
        if missing:
            warnings.append(
                f"Mode shapes unavailable for modes {missing}; frequencies remain available."
            )

        # Build flat displacement arrays for each mode shape
        for m_idx, shape in mode_shapes.items():
            flat = np.zeros(3 * n_points, dtype=float)
            flat[0::3] = shape[:, 0]
            flat[1::3] = shape[:, 1]
            flat[2::3] = shape[:, 2]
            mode_displacements_flat[m_idx] = flat

    n_points = int(np.asarray(mesh.p).shape[1])
    active_m = int(np.clip(active_mode, 1, max(1, len(modes) or 1)))
    active_disp_flat = mode_displacements_flat.get(
        active_m, np.zeros(3 * n_points, dtype=float)
    )
    active_shape = mode_shapes.get(active_m, np.zeros((n_points, 3), dtype=float))
    peak_disp = (
        float(np.max(np.linalg.norm(active_shape, axis=1)))
        if active_shape.size
        else 0.0
    )

    # Auto deformation scale
    bbox = np.ptp(np.asarray(mesh.p), axis=1)
    char_len = max(float(np.max(bbox)), 1e-9)
    auto_scale = (0.05 * char_len / peak_disp) if peak_disp > 1e-9 else 1.0
    # Eigenvectors are mass normalized; their amplitudes have arbitrary units.
    # Do not impose a physical-displacement lower bound on the visual scale.

    resolved_scale = _resolve_deformation_scale(deformation_scale, auto_scale, warnings)

    freq_hz = (
        modes[active_m - 1]["frequency_hz"]
        if (modes and active_m <= len(modes))
        else 0.0
    )

    return {
        "type": "modal",
        "backend": "CalculiX",
        "analysis_type": "Modal / Natural Frequencies",
        "input_deck": str(inp_path),
        "mesh": mesh,
        "external_status": status,
        "work_dir": str(work_dir),
        "solver_executable": executable,
        "solver_elapsed_s": solver_elapsed_s,
        "num_modes": len(modes),
        "modes": modes,
        "frequencies_hz": [m["frequency_hz"] for m in modes],
        "frequencies_rad": [m["frequency_rad_s"] for m in modes],
        "active_mode": active_m,
        "mode_shapes": mode_shapes,
        "mode_displacements_flat": mode_displacements_flat,
        "displacement": active_disp_flat,
        "peak_displacement": peak_disp,
        "frequency_hz": freq_hz,
        "auto_deformation_scale": auto_scale,
        "deformation_scale": resolved_scale,
        "visualization_mode": f"Mode {active_m} ({freq_hz:.1f} Hz)",
        "warnings": warnings,
    }
