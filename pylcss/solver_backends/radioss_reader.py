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
        try:
            subprocess.run(
                [converter, str(anim)],
                cwd=str(anim.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
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


def _vtk_point_data(mesh) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract displacement (N,3) and Von Mises (N,) from a meshio object.

    OpenRadioss's ``anim_to_vtk`` writes displacement under names that vary by
    release ("Displacement", "DISP", "DEPLACEMENT"); stress is most commonly
    "Stress" or "VonMises".  We probe the common spellings.
    """
    disp: Optional[np.ndarray] = None
    vm: Optional[np.ndarray] = None

    point_data = getattr(mesh, "point_data", {}) or {}
    for key in point_data:
        kl = key.lower()
        if disp is None and ("disp" in kl or "depla" in kl):
            arr = np.asarray(point_data[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                disp = arr[:, :3]
            elif arr.ndim == 1 and arr.size % 3 == 0:
                disp = arr.reshape((-1, 3))
        if vm is None and ("von" in kl or "vonmis" in kl or kl == "vm"):
            arr = np.asarray(point_data[key], dtype=float)
            if arr.ndim == 1:
                vm = arr
            elif arr.ndim == 2 and arr.shape[1] == 1:
                vm = arr[:, 0]
        if vm is None and kl in ("stress", "sigma", "p"):  # tensor → derive VM
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
    return disp, vm


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
    for vtk_idx, vtk_path in enumerate(vtk_paths):
        try:
            mesh = meshio.read(str(vtk_path))
        except Exception as exc:
            warnings.append(f"Failed to read {vtk_path.name}: {exc}")
            continue
        disp, vm = _vtk_point_data(mesh)
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
                "time": float(vtk_idx + 1) / float(n_anim),
            }
        )

    if not frames:
        warnings.append("All converted VTK files failed to parse; no frames produced.")
    return frames, warnings
