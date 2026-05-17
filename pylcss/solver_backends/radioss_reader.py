# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""OpenRadioss animation result reader.

OpenRadioss writes proprietary ``<run>A001``, ``<run>A002`` ... binary animation
files.  Reimplementing that format would be brittle; instead we rely on the
``anim_to_vtk`` converter that ships in the OpenRadioss tools/ directory, and
ingest the resulting ``.vtk`` files via meshio (already a PyLCSS dependency).

The animation frames are turned into the dict shape consumed by the crash
viewer (``frames`` with ``displacement``, ``stress_vm``, ``time``).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pylcss.solver_backends.common import resolve_executable


_ANIM_FILE_RE = re.compile(r"A(\d{3,4})$", re.IGNORECASE)


def find_animation_files(work_dir: str | Path, job_name: str) -> List[Path]:
    """Return the ``<job>A001``... animation files sorted by frame index."""
    work_dir = Path(work_dir)
    candidates: List[Tuple[int, Path]] = []
    for child in work_dir.iterdir():
        if not child.is_file():
            continue
        if not child.name.startswith(job_name):
            continue
        match = _ANIM_FILE_RE.search(child.name)
        if not match:
            continue
        try:
            idx = int(match.group(1))
        except ValueError:
            continue
        candidates.append((idx, child))
    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def resolve_anim_to_vtk(explicit: Optional[str] = None) -> Optional[str]:
    """Locate an ``anim_to_vtk`` converter binary if installed."""
    return resolve_executable(
        explicit,
        env_vars=("PYLCSS_OPENRADIOSS_ANIM2VTK", "OPENRADIOSS_ANIM2VTK"),
        candidates=(
            "anim_to_vtk",
            "anim_to_vtk.exe",
            "anim_to_vtk_win64.exe",
            "anim_to_vtk_linux64_gf",
            "anim_to_vtk_linux64",
            "anim_to_vtk_linux64_gf_sp",
            "anim_to_vtk_linux64_gf_dp",
        ),
    )


def convert_anim_files(
    anim_files: List[Path],
    converter: str,
    timeout_s: float = 600.0,
    max_workers: Optional[int] = None,
) -> List[Path]:
    """Run ``anim_to_vtk`` for each animation file in parallel.

    Each invocation is an independent process — ``ThreadPoolExecutor`` lets
    Python wait on multiple subprocesses concurrently, which is what we want.
    Using all logical cores cuts the 80-frame Crashbox conversion from ~90 s
    sequential down to ~10–15 s on commodity hardware.

    Defaults ``max_workers`` to ``min(os.cpu_count() or 4, 8)`` — more workers
    just thrash disk IO without speeding anything up on typical SSDs.
    """
    import concurrent.futures as cf

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)

    n = len(anim_files)
    if n == 0:
        return []
    print(f"OpenRadioss: converting {n} animation file(s) via anim_to_vtk "
          f"(parallelism={max_workers})...")

    def _convert_one(anim: Path) -> Path:
        # The OpenRadioss ``anim_to_vtk`` tool writes the converted VTK ASCII
        # to STDOUT (not to a file), so we must capture stdout and write it
        # to ``<anim>.vtk`` ourselves.  Earlier versions of this function
        # piped stdout to a dropped pipe, which is why the conversion appeared
        # to succeed (exit 0) but produced no .vtk files.
        try:
            proc = subprocess.run(
                [converter, str(anim)],
                cwd=str(anim.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return anim
        if proc.returncode == 0 and proc.stdout:
            out_path = anim.with_name(anim.name + ".vtk")
            try:
                out_path.write_text(proc.stdout, encoding="utf-8")
            except OSError:
                pass
        return anim

    done = 0
    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for _ in pool.map(_convert_one, anim_files):
            done += 1
            # Coarse progress so the user sees movement on long batches.
            if done == 1 or done == n or done % max(1, n // 10) == 0:
                print(f"  converted {done}/{n}")

    # Collect everything the converter dropped next to its inputs.
    vtk_paths: List[Path] = []
    seen: set = set()
    for anim in anim_files:
        for pattern in (anim.name + "*.vtk", anim.stem + "*.vtk"):
            for produced in sorted(anim.parent.glob(pattern)):
                if produced.is_file() and produced not in seen:
                    vtk_paths.append(produced)
                    seen.add(produced)
    return vtk_paths


def _von_mises_from_tensor(arr: np.ndarray) -> Optional[np.ndarray]:
    """Return Von Mises stress from common tensor layouts."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 3 and arr.shape[1] >= 3 and arr.shape[2] >= 3:
        sxx = arr[:, 0, 0]
        syy = arr[:, 1, 1]
        szz = arr[:, 2, 2]
        sxy = 0.5 * (arr[:, 0, 1] + arr[:, 1, 0])
        syz = 0.5 * (arr[:, 1, 2] + arr[:, 2, 1])
        szx = 0.5 * (arr[:, 2, 0] + arr[:, 0, 2])
    elif arr.ndim == 2 and arr.shape[1] >= 6:
        sxx, syy, szz = arr[:, 0], arr[:, 1], arr[:, 2]
        sxy, syz, szx = arr[:, 3], arr[:, 4], arr[:, 5]
    else:
        return None
    return np.sqrt(
        0.5
        * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
        )
    )


def _cell_vm_to_point_vm(mesh) -> Optional[np.ndarray]:
    """Average cell stress tensors onto VTK points when nodal VM is absent."""
    cell_data = getattr(mesh, "cell_data_dict", {}) or {}
    if not cell_data:
        return None

    n_points = int(mesh.points.shape[0]) if mesh.points is not None else 0
    if n_points <= 0:
        return None
    accum = np.zeros(n_points, dtype=float)
    counts = np.zeros(n_points, dtype=float)
    found = False

    # Sort field names so 3D-element Von Mises comes before 2D-element Von Mises.
    # OpenRadioss VTK files carry both "2DELEM_Von_Mises" and "3DELEM_Von_Mises";
    # iterating dict order would pick the 2D field first (all-zero for solid meshes),
    # set found=True, and break — leaving VM permanently zero.
    def _vm_sort_key(n: str) -> int:
        nl = n.lower()
        if "3delem" in nl and "von" in nl:
            return 0
        if "von" in nl or "stress" in nl:
            return 1
        return 2

    for name, by_type in sorted(cell_data.items(), key=lambda kv: _vm_sort_key(kv[0])):
        name_l = str(name).lower()
        if "stress" not in name_l and "von" not in name_l:
            continue
        for cell_type, values in by_type.items():
            arr = np.asarray(values, dtype=float)
            if "von" in name_l and arr.ndim >= 1:
                vm = arr.reshape(arr.shape[0], -1)[:, 0]
            else:
                vm = _von_mises_from_tensor(arr)
            if vm is None or vm.size == 0:
                continue
            blocks = [block.data for block in mesh.cells if block.type == cell_type]
            if not blocks:
                continue
            offset = 0
            for conn in blocks:
                n_cells = int(conn.shape[0])
                block_vm = vm[offset : offset + n_cells]
                offset += n_cells
                if block_vm.size != n_cells:
                    continue
                flat_conn = np.asarray(conn, dtype=int).reshape(-1)
                repeated = np.repeat(block_vm, conn.shape[1])
                np.add.at(accum, flat_conn, repeated)
                np.add.at(counts, flat_conn, 1.0)
                found = True
        if found:
            break

    if not found:
        return None
    vm_points = np.zeros(n_points, dtype=float)
    valid = counts > 0
    vm_points[valid] = accum[valid] / counts[valid]
    return vm_points


def _vtk_point_data(mesh) -> Tuple[
    Optional[np.ndarray],     # disp (N, 3)
    Optional[np.ndarray],     # vm   (N,)
    Optional[np.ndarray],     # node_ids (N,) 1-based
    Optional[np.ndarray],     # velocity (N, 3)
    Optional[np.ndarray],     # cell internal energy density per element (N_elem,)
]:
    """Extract displacement (N,3), Von Mises (N,), velocity (N,3), and cell
    internal energy density (N_elem,) from a meshio object.

    OpenRadioss's ``anim_to_vtk`` writes displacement under names that vary by
    release ("Displacement", "DISP", "DEPLACEMENT"); velocity is "VEL" /
    "Velocity" / "VITESSE"; stress is "Stress" or "VonMises"; internal-energy
    density is in cell_data under "Energy", "ENER", or "Internal_Energy".
    We probe the common spellings.
    """
    disp: Optional[np.ndarray] = None
    vm: Optional[np.ndarray] = None
    node_ids: Optional[np.ndarray] = None
    vel: Optional[np.ndarray] = None

    point_data = getattr(mesh, "point_data", {}) or {}
    for key in point_data:
        kl = key.lower()
        if node_ids is None and kl in ("node_id", "nodeid", "node ids", "node_ids"):
            node_ids = np.asarray(point_data[key], dtype=int).reshape(-1)
        if disp is None and ("disp" in kl or "depla" in kl):
            arr = np.asarray(point_data[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                disp = arr[:, :3]
            elif arr.ndim == 1 and arr.size % 3 == 0:
                disp = arr.reshape((-1, 3))
        if vel is None and (kl == "vel" or "veloc" in kl or "vitesse" in kl):
            arr = np.asarray(point_data[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                vel = arr[:, :3]
            elif arr.ndim == 1 and arr.size % 3 == 0:
                vel = arr.reshape((-1, 3))
        if vm is None and ("von" in kl or "vonmis" in kl or kl == "vm"):
            arr = np.asarray(point_data[key], dtype=float)
            if arr.ndim == 1:
                vm = arr
            elif arr.ndim == 2 and arr.shape[1] == 1:
                vm = arr[:, 0]
        if vm is None and kl in ("stress", "sigma", "p"):  # tensor to VM
            arr = np.asarray(point_data[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 6:
                sxx, syy, szz = arr[:, 0], arr[:, 1], arr[:, 2]
                sxy, syz, szx = arr[:, 3], arr[:, 4], arr[:, 5]
                vm = np.sqrt(
                    0.5
                    * (
                        (sxx - syy) ** 2
                        + (syy - szz) ** 2
                        + (szz - sxx) ** 2
                        + 6.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
                    )
                )
    if vm is None:
        vm = _cell_vm_to_point_vm(mesh)
    ener_cell = _extract_cell_ener(mesh)
    return disp, vm, node_ids, vel, ener_cell


def _extract_cell_ener(mesh) -> Optional[np.ndarray]:
    """Return per-element internal-energy density (one scalar per solid cell).

    OpenRadioss tags 2D vs 3D fields in cell_data — for tet meshes the 3D
    block is what we want.  Falls back to the first matching field if no
    explicit 3D tag is present.
    """
    cell_data = getattr(mesh, "cell_data_dict", {}) or {}
    if not cell_data:
        return None

    def _sort_key(n: str) -> int:
        nl = n.lower()
        if "3delem" in nl and "ener" in nl:
            return 0
        if "ener" in nl or "internal" in nl:
            return 1
        return 2

    for name, by_type in sorted(cell_data.items(), key=lambda kv: _sort_key(kv[0])):
        nl = name.lower()
        if "ener" not in nl and "internal" not in nl:
            continue
        # Concatenate all cell types preserving the meshio cell-block order
        # so indices line up with mesh.cells.
        out_blocks: List[np.ndarray] = []
        for block in mesh.cells:
            vals = by_type.get(block.type)
            if vals is None:
                out_blocks.append(np.zeros(block.data.shape[0], dtype=float))
                continue
            arr = np.asarray(vals, dtype=float)
            if arr.ndim > 1:
                arr = arr.reshape(arr.shape[0], -1)[:, 0]
            out_blocks.append(arr)
        if out_blocks:
            return np.concatenate(out_blocks)
    return None


def read_animation_frames(
    work_dir: str | Path,
    job_name: str,
    converter: Optional[str] = None,
    timeout_s: float = 600.0,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Build a list of viewer-ready crash frames from OpenRadioss output.

    Returns
    -------
    frames : list of dict
        ``{displacement, stress_vm, time}`` per frame, ordered by time.
    warnings : list of str
        Human-readable diagnostics suitable for the result dict's
        ``warnings`` array.
    """
    warnings: List[str] = []
    work_dir = Path(work_dir)
    anim_files = find_animation_files(work_dir, job_name)
    if not anim_files:
        warnings.append(
            "OpenRadioss produced no animation files. The Engine run may have "
            "stopped early, or *DATABASE_BINARY_D3PLOT was not respected."
        )
        return [], warnings

    converter_path = converter or resolve_anim_to_vtk()
    if not converter_path:
        warnings.append(
            "OpenRadioss animation files were generated but no anim_to_vtk "
            "converter was found.  Set PYLCSS_OPENRADIOSS_ANIM2VTK, drop the "
            "anim_to_vtk binary on PATH, or open the .A### files in HyperView / "
            "ParaView for visualization."
        )
        return [], warnings

    try:
        import meshio  # noqa: WPS433 — runtime import keeps PyLCSS startup fast.
    except Exception as exc:  # pragma: no cover - meshio is a hard dep.
        warnings.append(f"meshio import failed: {exc}; cannot ingest VTK frames.")
        return [], warnings

    vtk_paths = convert_anim_files(anim_files, converter_path, timeout_s=timeout_s)
    if not vtk_paths:
        warnings.append(
            "anim_to_vtk did not produce any .vtk files. Check converter version "
            "compatibility with the OpenRadioss release you have installed."
        )
        return [], warnings

    frames: List[Dict[str, object]] = []
    n_anim = max(len(anim_files), 1)
    max_disp_seen = 0.0
    max_vm_seen = 0.0
    for vtk_idx, vtk_path in enumerate(vtk_paths):
        try:
            mesh = meshio.read(str(vtk_path))
        except Exception as exc:
            warnings.append(f"Failed to read {vtk_path.name}: {exc}")
            continue
        disp, vm, node_ids, vel, ener_cell = _vtk_point_data(mesh)
        n_points = int(mesh.points.shape[0]) if mesh.points is not None else 0
        if disp is None:
            disp = np.zeros((n_points, 3), dtype=float)
        if vm is None:
            vm = np.zeros(n_points, dtype=float)
        # Flatten to the viewer's [3*N] layout (per-node X,Y,Z grouped).
        flat_disp = np.zeros(3 * n_points, dtype=float)
        flat_disp[0::3] = disp[:, 0]
        flat_disp[1::3] = disp[:, 1]
        flat_disp[2::3] = disp[:, 2]
        frames.append(
            {
                "displacement": flat_disp,
                "stress_vm": np.asarray(vm, dtype=float),
                "velocity": (np.asarray(vel, dtype=float) if vel is not None
                             else np.zeros((n_points, 3), dtype=float)),
                "ener_cell": (np.asarray(ener_cell, dtype=float) if ener_cell is not None
                              else None),
                "node_ids": node_ids,
                "time": float(vtk_idx + 1) / float(n_anim),
            }
        )
        max_disp_seen = max(max_disp_seen, float(np.max(np.abs(disp))) if disp.size else 0.0)
        max_vm_seen = max(max_vm_seen, float(np.max(vm)) if vm.size else 0.0)
    print(f"OpenRadioss frames: parsed {len(frames)} VTK files, "
          f"global max |u| = {max_disp_seen:.4e} mm, "
          f"global max |VM| raw = {max_vm_seen:.4e}")
    if max_disp_seen < 1e-6 and max_vm_seen < 1e-6:
        warnings.append(
            "All animation frames carry essentially zero displacement and stress.  "
            "This usually means the impact velocity is far below yield onset for "
            "the material+geometry combo — the simulation is purely elastic and "
            "the deformation is microscopic relative to the box size.  Raise the "
            "ImpactCondition velocity, or set disp_scale on the CrashSolver node "
            "to a large value (e.g. 1000) to amplify the elastic vibration for "
            "visualization."
        )

    if not frames:
        warnings.append("All converted VTK files failed to parse; no frames produced.")
    return frames, warnings
