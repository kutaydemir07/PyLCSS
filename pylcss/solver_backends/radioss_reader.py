# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""OpenRadioss animation result reader.

OpenRadioss writes proprietary ``<run>A001``, ``<run>A002`` ... binary animation
files.  Reimplementing that format would be brittle; instead we rely on the
``anim_to_vtk`` converter that ships in the OpenRadioss tools/ directory, and
ingest the resulting ``.vtk`` files via meshio (already a PyLCSS dependency).

The animation frames are turned into the dict shape consumed by the crash
viewer, including displacement, stress, plastic strain, erosion/failure,
velocity, energy, topology, and physical time.
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
_VTK_TIME_RE = re.compile(
    r"(?mi)^TIME\s+1\s+1\s+\w+\s*\r?\n\s*([-+0-9.eE]+)"
)


class RadiossAnimationMesh:
    """Small, pickle-safe mesh adapter used by crash playback.

    Existing user decks can contain mixed shells, solids, rigid-wall quads,
    and beams.  A scikit-fem mesh cannot represent that mixture, so retain the
    meshio cell blocks for the viewer while exposing ``p``/``t`` for legacy
    code and result export.
    """

    def __init__(self, points, cell_blocks):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Animation points must be (N, 3), got {pts.shape!r}.")
        self.p = np.ascontiguousarray(pts[:, :3].T)
        self.cell_blocks = []
        for cell_type, data in cell_blocks:
            conn = np.asarray(data, dtype=int)
            if conn.ndim == 2 and conn.size:
                self.cell_blocks.append((str(cell_type), np.ascontiguousarray(conn)))

        # Compatibility view for exporters and older viewer paths.  Prefer the
        # deformable solid/shell topology over rigid-wall or line cells.
        priority = (
            "tetra", "tetra10", "hexahedron", "wedge", "pyramid",
            "triangle", "quad", "line", "vertex",
        )
        primary_type = None
        primary_blocks = []
        for candidate in priority:
            blocks = [data for kind, data in self.cell_blocks if kind == candidate]
            if blocks:
                primary_type = candidate
                primary_blocks = blocks
                break
        self.cell_type = primary_type or "unknown"
        if primary_blocks:
            primary = np.concatenate(primary_blocks, axis=0)
            self.t = np.ascontiguousarray(primary.T)
            offsets = []
            offset = 0
            for kind, data in self.cell_blocks:
                if kind == primary_type:
                    offsets.extend(range(offset, offset + data.shape[0]))
                offset += data.shape[0]
            self.primary_cell_indices = np.asarray(offsets, dtype=int)
        else:
            self.t = np.empty((0, 0), dtype=int)
            self.primary_cell_indices = np.empty(0, dtype=int)

    @classmethod
    def from_meshio(cls, mesh):
        blocks = [(block.type, block.data) for block in (mesh.cells or [])]
        return cls(mesh.points, blocks)


def _read_vtk_time(path: Path) -> Optional[float]:
    """Read the physical animation time written in VTK FIELD metadata."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            header = stream.read(4096)
    except OSError:
        return None
    match = _VTK_TIME_RE.search(header)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


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


def _normalise_anim_vtk(text: str) -> str:
    """Fix anim_to_vtk's shell-element output so meshio's VTK reader accepts it.

    For *ELEMENT_SHELL triangles the OpenRadioss converter emits the LS-DYNA
    degenerate-quad convention in the ``CELLS`` block — four node indices with
    the fourth equal to the third — but tags the cell as ``VTK_TRIANGLE`` (5)
    in ``CELL_TYPES``.  meshio rejects this with "Couldn't read file ... as
    vtk" because a VTK triangle must list exactly three vertices.  We rewrite
    the CELLS block so every triangle row uses ``3 n1 n2 n3`` and adjust the
    cell-block size accordingly; non-triangle cells are left untouched.
    """
    lines = text.splitlines()
    try:
        i_cells = next(
            i for i, ln in enumerate(lines) if ln.startswith("CELLS ")
        )
    except StopIteration:
        return text
    try:
        n_cells_s, _ = lines[i_cells].split()[1:3]
        n_cells = int(n_cells_s)
    except (ValueError, IndexError):
        return text
    try:
        i_types = next(
            i for i, ln in enumerate(lines)
            if ln.startswith("CELL_TYPES ")
        )
    except StopIteration:
        return text

    cell_rows = lines[i_cells + 1 : i_cells + 1 + n_cells]
    type_rows = lines[i_types + 1 : i_types + 1 + n_cells]
    if len(cell_rows) != n_cells or len(type_rows) != n_cells:
        return text

    new_rows: list[str] = []
    total = 0
    rewritten = 0
    for row, type_row in zip(cell_rows, type_rows):
        try:
            cell_type = int(type_row.strip())
        except ValueError:
            new_rows.append(row)
            total += len(row.split())
            continue
        parts = row.split()
        if cell_type == 5 and len(parts) >= 4 and int(parts[0]) > 3:
            # Drop the trailing duplicated node and reset the leading count.
            keep = parts[1:4]
            row = "3 " + " ".join(keep)
            rewritten += 1
            total += 4
        else:
            total += len(parts)
        new_rows.append(row)

    if rewritten == 0:
        return text
    lines[i_cells] = f"CELLS {n_cells} {total}"
    lines[i_cells + 1 : i_cells + 1 + n_cells] = new_rows
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


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
                out_path.write_text(_normalise_anim_vtk(proc.stdout),
                                    encoding="utf-8")
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


def _legacy_cell_vm_to_point_vm(mesh) -> Optional[np.ndarray]:
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


def _extract_cell_scalar(mesh, name_matches, reducer: str = "max") -> Optional[np.ndarray]:
    """Return one scalar per meshio cell across all matching result fields."""
    blocks = list(getattr(mesh, "cells", []) or [])
    cell_data = getattr(mesh, "cell_data", {}) or {}
    if not blocks or not cell_data:
        return None
    block_sizes = [int(block.data.shape[0]) for block in blocks]
    total = sum(block_sizes)
    combined = None

    for name, values_by_block in cell_data.items():
        if not name_matches(str(name).lower()):
            continue
        candidate = np.zeros(total, dtype=float)
        offset = 0
        valid_field = False
        for block_index, n_cells in enumerate(block_sizes):
            values = values_by_block[block_index] if block_index < len(values_by_block) else None
            if values is not None:
                arr = np.asarray(values, dtype=float)
                if arr.size:
                    arr = arr.reshape(arr.shape[0], -1)[:, 0]
                    n_copy = min(n_cells, arr.size)
                    candidate[offset : offset + n_copy] = arr[:n_copy]
                    valid_field = True
            offset += n_cells
        if not valid_field:
            continue
        if combined is None:
            combined = candidate
        elif reducer == "min":
            combined = np.minimum(combined, candidate)
        elif reducer != "first":
            combined = np.maximum(combined, candidate)
    return combined


def _cell_scalar_to_points(mesh, values, reducer: str = "average") -> Optional[np.ndarray]:
    """Project block-aligned cell scalars onto their incident VTK points."""
    if values is None or mesh.points is None:
        return None
    n_points = int(mesh.points.shape[0])
    values = np.asarray(values, dtype=float).reshape(-1)
    if n_points <= 0 or values.size == 0:
        return None

    out = np.zeros(n_points, dtype=float)
    counts = np.zeros(n_points, dtype=float)
    offset = 0
    for block in (mesh.cells or []):
        conn = np.asarray(block.data, dtype=int)
        n_cells = int(conn.shape[0]) if conn.ndim == 2 else 0
        block_values = values[offset : offset + n_cells]
        offset += n_cells
        if n_cells == 0 or block_values.size != n_cells:
            continue
        flat_conn = conn.reshape(-1)
        repeated = np.repeat(block_values, conn.shape[1])
        valid = (flat_conn >= 0) & (flat_conn < n_points)
        if reducer == "max":
            np.maximum.at(out, flat_conn[valid], repeated[valid])
        else:
            np.add.at(out, flat_conn[valid], repeated[valid])
            np.add.at(counts, flat_conn[valid], 1.0)
    if reducer != "max":
        valid = counts > 0
        out[valid] /= counts[valid]
    return out


# Override the legacy solid-first implementation above. OpenRadioss writes
# parallel 2D/3D arrays with zeros in the irrelevant field; combining both is
# required for shell and mixed-element decks.
def _cell_vm_to_point_vm(mesh) -> Optional[np.ndarray]:
    vm_cell = _extract_cell_scalar(
        mesh,
        lambda name: "von" in name or "vonmis" in name,
        reducer="max",
    )
    if vm_cell is not None:
        return _cell_scalar_to_points(mesh, vm_cell, reducer="average")

    for name, values_by_block in (getattr(mesh, "cell_data", {}) or {}).items():
        if "stress" not in str(name).lower():
            continue
        tensors = []
        for values in values_by_block:
            vm = _von_mises_from_tensor(np.asarray(values, dtype=float))
            if vm is not None:
                tensors.append(vm)
        if tensors:
            return _cell_scalar_to_points(
                mesh, np.concatenate(tensors), reducer="average"
            )
    return None


def _vtk_point_data(mesh) -> Tuple[
    Optional[np.ndarray],     # disp (N, 3)
    Optional[np.ndarray],     # vm   (N,)
    Optional[np.ndarray],     # node_ids (N,) 1-based
    Optional[np.ndarray],     # velocity (N, 3)
    Optional[np.ndarray],     # cell internal energy density per element (N_elem,)
    Optional[np.ndarray],     # equivalent plastic strain per point (N,)
    Optional[np.ndarray],     # failed flag per point (N,)
    Optional[np.ndarray],     # element ids (N_elem,) 1-based, 0 for generated cells
    Optional[np.ndarray],     # equivalent plastic strain per cell (N_elem,)
    Optional[np.ndarray],     # failed flag per cell (N_elem,)
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
    eps_cell = _extract_cell_scalar(
        mesh,
        lambda name: "plastic" in name and "strain" in name,
        reducer="max",
    )
    erosion_status = _extract_cell_scalar(
        mesh,
        lambda name: "erosion" in name and "status" in name,
        reducer="min",
    )
    failed_cell = None
    if erosion_status is not None:
        # anim_to_vtk writes 1 for active elements and 0 for deleted elements.
        failed_cell = (np.asarray(erosion_status) <= 0.5).astype(float)
    eps_point = _cell_scalar_to_points(mesh, eps_cell, reducer="max")
    failed_point = _cell_scalar_to_points(mesh, failed_cell, reducer="max")
    element_ids = _extract_cell_scalar(
        mesh,
        lambda name: name in ("element_id", "elementid", "element ids", "element_ids"),
        reducer="first",
    )
    if element_ids is not None:
        element_ids = np.asarray(element_ids, dtype=int)
    return (
        disp, vm, node_ids, vel, ener_cell, eps_point, failed_point,
        element_ids, eps_cell, failed_cell,
    )


def _extract_cell_ener(mesh) -> Optional[np.ndarray]:
    """Return per-element internal-energy density (one scalar per solid cell).

    OpenRadioss tags 2D vs 3D fields in cell_data — for tet meshes the 3D
    block is what we want.  Falls back to the first matching field if no
    explicit 3D tag is present.
    """
    combined = _extract_cell_scalar(
        mesh,
        lambda name: "ener" in name or "internal" in name,
        reducer="max",
    )
    if combined is not None:
        return combined

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
    end_time: Optional[float] = None,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Build a list of viewer-ready crash frames from OpenRadioss output.

    Returns
    -------
    frames : list of dict
        Viewer-ready field dictionaries per frame, ordered by physical time.
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
            "converter was found. Set PYLCSS_OPENRADIOSS_ANIM2VTK or open the "
            ".A### files in HyperView / ParaView for visualization."
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
    viewer_mesh = None
    for vtk_idx, vtk_path in enumerate(vtk_paths):
        try:
            mesh = meshio.read(str(vtk_path))
        except Exception as exc:
            warnings.append(f"Failed to read {vtk_path.name}: {exc}")
            continue
        if viewer_mesh is None:
            try:
                viewer_mesh = RadiossAnimationMesh.from_meshio(mesh)
            except Exception as exc:
                warnings.append(f"Failed to build playback mesh from {vtk_path.name}: {exc}")

        (
            disp, vm, node_ids, vel, ener_cell, eps_point, failed_point,
            element_ids, eps_cell, failed_cell,
        ) = _vtk_point_data(mesh)
        vm_cell = _extract_cell_scalar(
            mesh,
            lambda name: "von" in name or "vonmis" in name,
            reducer="max",
        )
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
        physical_time = _read_vtk_time(vtk_path)
        time_is_normalized = False
        if physical_time is None:
            fraction = float(vtk_idx + 1) / float(n_anim)
            if end_time is not None:
                physical_time = fraction * float(end_time)
            else:
                physical_time = fraction
                time_is_normalized = True

        frames.append(
            {
                "mesh": viewer_mesh,
                "displacement": flat_disp,
                "stress_vm": np.asarray(vm, dtype=float),
                "stress_vm_cell": (np.asarray(vm_cell, dtype=float)
                                   if vm_cell is not None else None),
                "velocity": (np.asarray(vel, dtype=float) if vel is not None
                              else np.zeros((n_points, 3), dtype=float)),
                "ener_cell": (np.asarray(ener_cell, dtype=float) if ener_cell is not None
                               else None),
                "node_ids": node_ids,
                "element_ids": element_ids,
                "eps_p": (np.asarray(eps_point, dtype=float) if eps_point is not None
                          else np.zeros(n_points, dtype=float)),
                "failed": (np.asarray(failed_point, dtype=float) if failed_point is not None
                           else np.zeros(n_points, dtype=float)),
                "eps_p_cell": (np.asarray(eps_cell, dtype=float) if eps_cell is not None
                               else None),
                "failed_cell": (np.asarray(failed_cell, dtype=float) if failed_cell is not None
                                else None),
                "time": float(physical_time),
                "time_is_normalized": time_is_normalized,
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
