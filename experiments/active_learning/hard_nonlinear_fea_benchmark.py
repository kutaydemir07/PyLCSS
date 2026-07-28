"""Hard-nonlinear CalculiX benchmark for PyLCSS active learning.

The production active-learning implementation is evaluated on a displacement-
controlled shallow arch that passes through a geometric snap-through regime.
The benchmark is deliberately separate from the smooth perforated-plate study:
it uses real CalculiX ``B31`` beam solves, preserves a never-revealed holdout
set, and compares GP-RF committee selection with a strong nested maximin design
at equal FEA budgets.

The expensive dataset is persistent and resumable.  During replay, labels are
revealed to the acquisition function only after their pool index is selected.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import qmc, ttest_rel, wilcoxon
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pylcss.solver_backends.calculix import resolve_calculix_executable
from pylcss.solver_backends.common import run_process, tail
from pylcss.surrogate_modeling.active_learning import (
    acquisition_scores,
    diverse_top_k,
    normalize_to_unit,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    REPO_ROOT
    / "experiments"
    / "active_learning"
    / "results"
    / "hard_nonlinear_snapthrough"
)
DEFAULT_DATASET_PATH = DEFAULT_RESULTS_DIR / "dataset.csv"

PARAMETER_NAMES = (
    "rise_mm",
    "thickness_mm",
    "displacement_ratio",
    "imperfection_ratio",
)
BOUNDS = (
    (6.0, 18.0),
    (1.0, 3.0),
    (0.10, 1.40),
    (-0.03, 0.03),
)
OUTPUT_NAMES = (
    "final_force_n",
    "pre_peak_force_n",
    "pre_peak_displacement_mm",
    "strain_energy_nmm",
)

HALF_SPAN_MM = 100.0
SECTION_WIDTH_MM = 12.0
YOUNGS_MODULUS_MPA = 210000.0
POISSONS_RATIO = 0.3
DEFAULT_MESH_ELEMENTS = 40
DEFAULT_MAX_INCREMENT = 0.02
DEFAULT_POOL_SIZE = 160
DEFAULT_TEST_SIZE = 64
DEFAULT_DATA_SEED = 20260727
DEFAULT_BENCHMARK_SEEDS = 20
DEFAULT_BUDGETS = (16, 24, 32, 48, 64)
ACTIVE_INITIAL = 12
ACTIVE_BATCH = 4
PRIMARY_BUDGET = 32
MINIMUM_MEAN_IMPROVEMENT_PCT = 5.0
MINIMUM_WIN_FRACTION = 0.60
REPLACEMENT_ACTIVE_BUDGET = 64
REPLACEMENT_REFERENCE_BUDGETS = (64, 72, 80, 88, 96, 100)
REPLACEMENT_AGGREGATE_MARGIN_PCT = 10.0
REPLACEMENT_OUTPUT_MARGIN_PCT = 15.0
REPLACEMENT_TRANSITION_MARGIN_PCT = 10.0
REPLACEMENT_MAX_NRMSE = 0.05
REPLACEMENT_MIN_R2 = 0.99

DATASET_FIELDS = (
    "sample_id",
    "split",
    *PARAMETER_NAMES,
    *OUTPUT_NAMES,
    "minimum_force_n",
    "minimum_tangent_n_per_mm",
    "regime_code",
    "snapthrough",
    "n_increments",
    "n_cutbacks",
    "elapsed_s",
    "mesh_elements",
    "max_increment",
    "solver_version",
    "solver_sha256",
    "success",
    "error",
)

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def design_points(
    pool_size: int = DEFAULT_POOL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_DATA_SEED,
) -> list[dict[str, object]]:
    """Create deterministic independent LHS pool and holdout designs."""

    if pool_size < 1 or test_size < 1:
        raise ValueError("pool_size and test_size must be positive.")
    bounds = np.asarray(BOUNDS, dtype=float)
    rows: list[dict[str, object]] = []
    for split, count, split_seed in (
        ("pool", pool_size, seed),
        ("test", test_size, seed + 1),
    ):
        unit = qmc.LatinHypercube(d=len(BOUNDS), seed=split_seed).random(count)
        physical = qmc.scale(unit, bounds[:, 0], bounds[:, 1])
        for index, point in enumerate(physical):
            rows.append(
                {
                    "sample_id": f"{split}_{index:03d}",
                    "split": split,
                    **{
                        name: float(value)
                        for name, value in zip(PARAMETER_NAMES, point)
                    },
                }
            )
    return rows


def _node_lines_for_arch(
    rise_mm: float,
    imperfection_ratio: float,
    mesh_elements: int,
) -> tuple[list[str], int]:
    if mesh_elements < 4 or mesh_elements % 2:
        raise ValueError("mesh_elements must be an even integer of at least four.")
    x = np.linspace(-HALF_SPAN_MM, HALF_SPAN_MM, mesh_elements + 1)
    phase = (x + HALF_SPAN_MM) / (2.0 * HALF_SPAN_MM)
    base = rise_mm * (1.0 - (x / HALF_SPAN_MM) ** 2)
    imperfection = imperfection_ratio * rise_mm * np.sin(2.0 * np.pi * phase)
    y = base + imperfection
    lines = [
        f"{index}, {xv:.12g}, {yv:.12g}, 0.0"
        for index, (xv, yv) in enumerate(zip(x, y), start=1)
    ]
    return lines, mesh_elements // 2 + 1


def build_shallow_arch_deck(
    design: Mapping[str, object],
    *,
    mesh_elements: int = DEFAULT_MESH_ELEMENTS,
    max_increment: float = DEFAULT_MAX_INCREMENT,
) -> tuple[str, int, float]:
    """Return a displacement-controlled B31 shallow-arch CalculiX deck."""

    rise = float(design["rise_mm"])
    thickness = float(design["thickness_mm"])
    displacement_ratio = float(design["displacement_ratio"])
    imperfection_ratio = float(design["imperfection_ratio"])
    if rise <= 0.0 or thickness <= 0.0 or displacement_ratio <= 0.0:
        raise ValueError("Arch rise, thickness, and displacement ratio must be positive.")
    if not 0.0 < max_increment <= 0.2:
        raise ValueError("max_increment must lie in (0, 0.2].")

    node_lines, apex_id = _node_lines_for_arch(
        rise, imperfection_ratio, mesh_elements
    )
    target_displacement = rise * displacement_ratio
    initial_increment = min(max_increment / 2.0, 0.01)
    element_lines = [
        f"{index}, {index}, {index + 1}"
        for index in range(1, mesh_elements + 1)
    ]
    lines = [
        "*HEADING",
        "PyLCSS hard nonlinear shallow-arch snap-through benchmark",
        "*NODE",
        *node_lines,
        "*ELEMENT, TYPE=B31, ELSET=ARCH",
        *element_lines,
        "*NSET, NSET=ALLNODES, GENERATE",
        f"1, {mesh_elements + 1}, 1",
        "*NSET, NSET=SUPPORTS",
        f"1, {mesh_elements + 1}",
        "*NSET, NSET=APEX",
        str(apex_id),
        "*MATERIAL, NAME=STEEL",
        "*ELASTIC",
        f"{YOUNGS_MODULUS_MPA:.12g}, {POISSONS_RATIO:.12g}",
        "*BEAM SECTION, ELSET=ARCH, MATERIAL=STEEL, SECTION=RECT",
        f"{SECTION_WIDTH_MM:.12g}, {thickness:.12g}",
        "0.0, 0.0, 1.0",
        "*BOUNDARY",
        "ALLNODES, 3, 5, 0.0",
        "SUPPORTS, 1, 2, 0.0",
        "APEX, 1, 1, 0.0",
        "*STEP, NLGEOM, INC=1000",
        "*STATIC",
        f"{initial_increment:.12g}, 1.0, 1e-6, {max_increment:.12g}",
        "*BOUNDARY",
        f"APEX, 2, 2, {-target_displacement:.12g}",
        "*NODE PRINT, NSET=APEX, FREQUENCY=1",
        "U, RF",
        "*END STEP",
        "",
    ]
    return "\n".join(lines), apex_id, target_displacement


def build_two_bar_deck(
    *,
    rise_mm: float,
    area_mm2: float,
    displacement_mm: float,
    max_increment: float = 0.02,
) -> tuple[str, int, float]:
    """Create the canonical two-bar von Mises truss verification deck."""

    lines = [
        "*HEADING",
        "PyLCSS analytical two-bar snap-through verification",
        "*NODE",
        f"1, {-HALF_SPAN_MM:.12g}, 0.0, 0.0",
        f"2, 0.0, {rise_mm:.12g}, 0.0",
        f"3, {HALF_SPAN_MM:.12g}, 0.0, 0.0",
        "*ELEMENT, TYPE=T3D2, ELSET=TRUSS",
        "1, 1, 2",
        "2, 2, 3",
        "*NSET, NSET=SUPPORTS",
        "1, 3",
        "*NSET, NSET=APEX",
        "2",
        "*MATERIAL, NAME=STEEL",
        "*ELASTIC",
        f"{YOUNGS_MODULUS_MPA:.12g}, {POISSONS_RATIO:.12g}",
        "*SOLID SECTION, ELSET=TRUSS, MATERIAL=STEEL",
        f"{area_mm2:.12g}",
        "*BOUNDARY",
        "SUPPORTS, 1, 3, 0.0",
        "APEX, 1, 1, 0.0",
        "APEX, 3, 3, 0.0",
        "*STEP, NLGEOM, INC=1000",
        "*STATIC",
        f"{min(0.01, max_increment / 2.0):.12g}, 1.0, 1e-6, {max_increment:.12g}",
        "*BOUNDARY",
        f"APEX, 2, 2, {-displacement_mm:.12g}",
        "*NODE PRINT, NSET=APEX, FREQUENCY=1",
        "U, RF",
        "*END STEP",
        "",
    ]
    return "\n".join(lines), 2, displacement_mm


def two_bar_analytical_force(
    *,
    rise_mm: float,
    area_mm2: float,
    displacement_mm: float,
) -> float:
    """Downward actuator force for a pin-jointed elastic two-bar truss."""

    initial_length = math.hypot(HALF_SPAN_MM, rise_mm)
    deformed_rise = rise_mm - displacement_mm
    current_length = math.hypot(HALF_SPAN_MM, deformed_rise)
    engineering_strain = (current_length - initial_length) / initial_length
    axial_force = YOUNGS_MODULUS_MPA * area_mm2 * engineering_strain
    reaction_y = 2.0 * axial_force * deformed_rise / current_length
    return -reaction_y


def _extract_nodal_blocks(
    text: str,
    heading: str,
    node_id: int,
) -> dict[float, np.ndarray]:
    lines = text.splitlines()
    result: dict[float, np.ndarray] = {}
    heading_lower = heading.lower()
    for index, line in enumerate(lines):
        if heading_lower not in line.lower() or " time " not in line.lower():
            continue
        match = re.search(rf"\btime\s+({_FLOAT})", line, flags=re.IGNORECASE)
        if not match:
            continue
        time_value = float(match.group(1).replace("D", "E").replace("d", "e"))
        for candidate in lines[index + 1 : index + 8]:
            tokens = candidate.split()
            if len(tokens) < 4:
                continue
            try:
                candidate_id = int(tokens[0])
            except ValueError:
                continue
            if candidate_id != node_id:
                continue
            values = np.asarray(
                [float(value.replace("D", "E").replace("d", "e")) for value in tokens[1:4]],
                dtype=float,
            )
            result[round(time_value, 10)] = values
            break
    return result


def parse_reaction_history(
    dat_path: Path,
    *,
    apex_id: int,
    target_displacement_mm: float,
    rise_mm: float,
) -> dict[str, object]:
    """Parse converged apex displacement/reaction history from a CCX .dat."""

    text = dat_path.read_text(encoding="utf-8", errors="replace")
    displacement = _extract_nodal_blocks(text, "displacements", apex_id)
    forces = _extract_nodal_blocks(text, "forces", apex_id)
    common_times = sorted(set(displacement) & set(forces))
    if not common_times:
        raise RuntimeError("CalculiX .dat contains no paired apex U/RF history.")

    down = np.asarray([-displacement[t][1] for t in common_times], dtype=float)
    actuator = np.asarray([-forces[t][1] for t in common_times], dtype=float)
    valid = np.isfinite(down) & np.isfinite(actuator)
    down, actuator = down[valid], actuator[valid]
    if len(down) < 3:
        raise RuntimeError("CalculiX returned fewer than three finite increments.")
    order = np.argsort(down, kind="stable")
    down, actuator = down[order], actuator[order]
    unique = np.concatenate([[True], np.diff(down) > 1e-10])
    down, actuator = down[unique], actuator[unique]
    down = np.concatenate([[0.0], down])
    actuator = np.concatenate([[0.0], actuator])

    tolerance = max(1e-5, 2e-4 * target_displacement_mm)
    if abs(float(down[-1]) - target_displacement_mm) > tolerance:
        raise RuntimeError(
            f"Final apex displacement {down[-1]:.8g} mm does not match "
            f"target {target_displacement_mm:.8g} mm."
        )

    grid = np.linspace(0.0, target_displacement_mm, 301)
    force_grid = np.interp(grid, down, actuator)
    tangent = np.gradient(force_grid, grid, edge_order=2)
    pre_mask = grid <= min(target_displacement_mm, 1.15 * rise_mm)
    pre_indices = np.flatnonzero(pre_mask)
    peak_local = int(pre_indices[np.argmax(force_grid[pre_mask])])
    peak_force = float(force_grid[peak_local])
    peak_displacement = float(grid[peak_local])
    minimum_force = float(np.min(force_grid))
    minimum_tangent = float(np.min(tangent))
    energy = float(np.trapezoid(force_grid, grid))

    transition_width = 0.10 * rise_mm
    peak_at_end = peak_local >= len(grid) - 3
    normalized_end_slope = (
        float(tangent[-1]) * rise_mm / max(abs(peak_force), 1.0)
    )
    if peak_at_end and normalized_end_slope > 0.05:
        regime_code = 0
    elif target_displacement_mm <= peak_displacement + transition_width:
        regime_code = 1
    else:
        regime_code = 2
    snapthrough = int(
        regime_code == 2
        and minimum_tangent < -0.05 * max(abs(peak_force), 1.0) / max(rise_mm, 1.0)
    )
    return {
        "time": np.asarray(common_times, dtype=float),
        "displacement_mm": down,
        "actuator_force_n": actuator,
        "final_force_n": float(force_grid[-1]),
        "pre_peak_force_n": peak_force,
        "minimum_force_n": minimum_force,
        "strain_energy_nmm": energy,
        "pre_peak_displacement_mm": peak_displacement,
        "minimum_tangent_n_per_mm": minimum_tangent,
        "regime_code": regime_code,
        "snapthrough": snapthrough,
    }


def _solver_version(executable: str) -> str:
    import subprocess

    proc = subprocess.run(
        [executable, "-v"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )
    match = re.search(r"Version\s+([0-9.]+)", proc.stdout or "")
    return match.group(1) if match else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sta_diagnostics(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) >= 7 and tokens[0].isdigit() and tokens[1].isdigit():
            rows.append(tokens)
    if not rows:
        return 0, 0
    increment_attempts: dict[int, int] = defaultdict(int)
    for row in rows:
        increment_attempts[int(row[1])] += 1
    return len(increment_attempts), sum(max(0, count - 1) for count in increment_attempts.values())


def solve_deck(
    deck: str,
    *,
    apex_id: int,
    target_displacement_mm: float,
    rise_mm: float,
    case_dir: Path,
    executable: str,
    job_name: str = "snapthrough",
    timeout_s: float = 120.0,
) -> dict[str, object]:
    """Run one persistent real CalculiX case and return parsed path metrics."""

    case_dir.mkdir(parents=True, exist_ok=True)
    inp_path = case_dir / f"{job_name}.inp"
    inp_path.write_text(deck, encoding="utf-8")
    started = time.perf_counter()
    proc = run_process(
        [executable, job_name],
        cwd=case_dir,
        timeout_s=timeout_s,
        extra_path_dirs=(str(Path(executable).resolve().parent),),
    )
    elapsed = time.perf_counter() - started
    (case_dir / "solver.log").write_text(
        tail(proc.stdout or "", 12000), encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CalculiX exited with {proc.returncode}: {tail(proc.stdout or '', 3000)}"
        )
    dat_path = case_dir / f"{job_name}.dat"
    if not dat_path.is_file():
        raise RuntimeError("CalculiX completed without a .dat reaction history.")
    history = parse_reaction_history(
        dat_path,
        apex_id=apex_id,
        target_displacement_mm=target_displacement_mm,
        rise_mm=rise_mm,
    )
    increments, cutbacks = _sta_diagnostics(case_dir / f"{job_name}.sta")
    history.update(
        {
            "elapsed_s": elapsed,
            "n_increments": increments,
            "n_cutbacks": cutbacks,
        }
    )
    return history


def _read_latest_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    latest: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id") or "")
            if sample_id:
                latest[sample_id] = row
    return latest


def _row_succeeded(row: Mapping[str, object] | None) -> bool:
    return bool(row) and str(row.get("success", "")).strip() == "1"


def _validate_resume_design(
    expected: Sequence[Mapping[str, object]],
    existing: Mapping[str, Mapping[str, str]],
) -> None:
    for design in expected:
        previous = existing.get(str(design["sample_id"]))
        if not previous:
            continue
        for name in PARAMETER_NAMES:
            if not math.isclose(
                float(previous[name]),
                float(design[name]),
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    f"Existing {design['sample_id']} does not match the requested "
                    "dataset definition. Use a new results directory."
                )


def generate_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_DATA_SEED,
    mesh_elements: int = DEFAULT_MESH_ELEMENTS,
    max_increment: float = DEFAULT_MAX_INCREMENT,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run or resume the real-CalculiX pool and untouched holdout dataset."""

    executable = resolve_calculix_executable()
    if not executable:
        raise FileNotFoundError("CalculiX executable was not found.")
    executable_path = Path(executable).resolve()
    solver_version = _solver_version(str(executable_path))
    solver_sha = _sha256(executable_path)
    expected = design_points(pool_size, test_size, seed)
    existing = _read_latest_rows(dataset_path)
    _validate_resume_design(expected, existing)
    pending = [
        design
        for design in expected
        if not _row_succeeded(existing.get(str(design["sample_id"])))
    ]
    if not pending:
        return {
            "completed": len(expected),
            "failed": 0,
            "pending": 0,
            "solver_version": solver_version,
            "solver_sha256": solver_sha,
        }

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    raw_root = results_dir / "raw_cases"
    raw_root.mkdir(parents=True, exist_ok=True)
    needs_header = not dataset_path.is_file() or dataset_path.stat().st_size == 0
    failures = 0
    with dataset_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
        if needs_header:
            writer.writeheader()
            handle.flush()
        for pending_index, design in enumerate(pending, start=1):
            output: dict[str, object] = {
                **design,
                **{name: "" for name in OUTPUT_NAMES},
                "minimum_force_n": "",
                "minimum_tangent_n_per_mm": "",
                "regime_code": "",
                "snapthrough": "",
                "n_increments": "",
                "n_cutbacks": "",
                "elapsed_s": "",
                "mesh_elements": mesh_elements,
                "max_increment": max_increment,
                "solver_version": solver_version,
                "solver_sha256": solver_sha,
                "success": 0,
                "error": "",
            }
            try:
                deck, apex_id, target = build_shallow_arch_deck(
                    design,
                    mesh_elements=mesh_elements,
                    max_increment=max_increment,
                )
                history = solve_deck(
                    deck,
                    apex_id=apex_id,
                    target_displacement_mm=target,
                    rise_mm=float(design["rise_mm"]),
                    case_dir=raw_root / str(design["sample_id"]),
                    executable=str(executable_path),
                )
                for name in (
                    *OUTPUT_NAMES,
                    "minimum_force_n",
                    "minimum_tangent_n_per_mm",
                    "regime_code",
                    "snapthrough",
                    "n_increments",
                    "n_cutbacks",
                    "elapsed_s",
                ):
                    output[name] = history[name]
                output["success"] = 1
            except Exception as exc:
                failures += 1
                output["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(output)
            handle.flush()
            progress(
                f"[{pending_index:03d}/{len(pending):03d}] "
                f"{design['sample_id']} h={float(design['rise_mm']):5.2f} "
                f"t={float(design['thickness_mm']):4.2f} "
                f"d/h={float(design['displacement_ratio']):4.2f} -> "
                + (
                    f"F={float(output['final_force_n']):9.3f} N, "
                    f"regime={output['regime_code']}"
                    if output["success"]
                    else f"FAILED: {output['error']}"
                )
            )

    latest = _read_latest_rows(dataset_path)
    completed = sum(_row_succeeded(latest.get(str(row["sample_id"]))) for row in expected)
    return {
        "completed": int(completed),
        "failed": failures,
        "pending": len(expected) - int(completed),
        "solver_version": solver_version,
        "solver_sha256": solver_sha,
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows were produced for {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative_difference(value: float, reference: float, scale: float = 1.0) -> float:
    denominator = max(abs(reference), abs(scale), 1e-12)
    return abs(value - reference) / denominator


def run_pilot(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Qualify solver correlation, mesh/increment stability, and nonlinearity."""

    executable = resolve_calculix_executable()
    if not executable:
        raise FileNotFoundError("CalculiX executable was not found.")
    executable = str(Path(executable).resolve())
    pilot_dir = results_dir / "pilot"
    raw_dir = pilot_dir / "raw_cases"
    raw_dir.mkdir(parents=True, exist_ok=True)

    analytical_rows: list[dict[str, object]] = []
    rise, area = 10.0, 10.0
    for ratio in (0.20, 0.45, 0.70, 1.00, 1.30, 1.60):
        target = ratio * rise
        deck, apex_id, target = build_two_bar_deck(
            rise_mm=rise,
            area_mm2=area,
            displacement_mm=target,
        )
        result = solve_deck(
            deck,
            apex_id=apex_id,
            target_displacement_mm=target,
            rise_mm=rise,
            case_dir=raw_dir / f"two_bar_{ratio:.2f}".replace(".", "p"),
            executable=executable,
            job_name="two_bar",
        )
        analytical = two_bar_analytical_force(
            rise_mm=rise,
            area_mm2=area,
            displacement_mm=target,
        )
        analytical_rows.append(
            {
                "displacement_ratio": ratio,
                "fea_force_n": float(result["final_force_n"]),
                "analytical_force_n": analytical,
                "absolute_error_n": abs(float(result["final_force_n"]) - analytical),
            }
        )
        progress(f"[pilot analytical] d/h={ratio:.2f}")
    analytical_fea = np.asarray([float(row["fea_force_n"]) for row in analytical_rows])
    analytical_truth = np.asarray(
        [float(row["analytical_force_n"]) for row in analytical_rows]
    )
    analytical_scale = max(float(np.max(np.abs(analytical_truth))), 1.0)
    analytical_nrmse = float(
        np.sqrt(np.mean((analytical_fea - analytical_truth) ** 2)) / analytical_scale
    )
    analytical_max_error = float(
        np.max(np.abs(analytical_fea - analytical_truth)) / analytical_scale
    )

    verification_designs = (
        ("pre_limit", {"rise_mm": 10.0, "thickness_mm": 1.8, "displacement_ratio": 0.30, "imperfection_ratio": 0.01}),
        ("near_limit", {"rise_mm": 10.0, "thickness_mm": 1.8, "displacement_ratio": 0.50, "imperfection_ratio": 0.01}),
        ("post_limit", {"rise_mm": 10.0, "thickness_mm": 1.8, "displacement_ratio": 1.30, "imperfection_ratio": 0.01}),
    )
    mesh_rows: list[dict[str, object]] = []
    for case_name, design in verification_designs:
        for mesh_elements in (20, 40, 80):
            deck, apex_id, target = build_shallow_arch_deck(
                design,
                mesh_elements=mesh_elements,
                max_increment=DEFAULT_MAX_INCREMENT,
            )
            result = solve_deck(
                deck,
                apex_id=apex_id,
                target_displacement_mm=target,
                rise_mm=float(design["rise_mm"]),
                case_dir=raw_dir / f"mesh_{case_name}_{mesh_elements}",
                executable=executable,
            )
            mesh_rows.append(
                {
                    "case": case_name,
                    "mesh_elements": mesh_elements,
                    **{name: float(result[name]) for name in OUTPUT_NAMES},
                    "n_increments": int(result["n_increments"]),
                    "n_cutbacks": int(result["n_cutbacks"]),
                }
            )
            progress(f"[pilot mesh] {case_name} n={mesh_elements}")

    mesh_differences = []
    for case_name, design in verification_designs:
        medium = next(
            row for row in mesh_rows
            if row["case"] == case_name and row["mesh_elements"] == 40
        )
        fine = next(
            row for row in mesh_rows
            if row["case"] == case_name and row["mesh_elements"] == 80
        )
        for output_name in OUTPUT_NAMES:
            scale = (
                float(design["rise_mm"])
                if output_name == "pre_peak_displacement_mm"
                else max(abs(float(fine[output_name])), abs(float(medium[output_name])), 1.0)
            )
            mesh_differences.append(
                {
                    "case": case_name,
                    "output": output_name,
                    "medium_vs_fine_relative": _relative_difference(
                        float(medium[output_name]), float(fine[output_name]), scale
                    ),
                }
            )
    mesh_max_relative = max(
        float(row["medium_vs_fine_relative"]) for row in mesh_differences
    )

    increment_design = dict(verification_designs[-1][1])
    increment_rows: list[dict[str, object]] = []
    for max_increment in (0.04, 0.02, 0.01):
        deck, apex_id, target = build_shallow_arch_deck(
            increment_design,
            mesh_elements=DEFAULT_MESH_ELEMENTS,
            max_increment=max_increment,
        )
        result = solve_deck(
            deck,
            apex_id=apex_id,
            target_displacement_mm=target,
            rise_mm=float(increment_design["rise_mm"]),
            case_dir=raw_dir / f"increment_{max_increment:.3f}".replace(".", "p"),
            executable=executable,
        )
        increment_rows.append(
            {
                "max_increment": max_increment,
                **{name: float(result[name]) for name in OUTPUT_NAMES},
                "n_increments": int(result["n_increments"]),
                "n_cutbacks": int(result["n_cutbacks"]),
            }
        )
        progress(f"[pilot increment] max={max_increment:.3f}")
    medium_increment = next(row for row in increment_rows if row["max_increment"] == 0.02)
    fine_increment = next(row for row in increment_rows if row["max_increment"] == 0.01)
    increment_differences = []
    for output_name in OUTPUT_NAMES:
        scale = (
            float(increment_design["rise_mm"])
            if output_name == "pre_peak_displacement_mm"
            else max(
                abs(float(medium_increment[output_name])),
                abs(float(fine_increment[output_name])),
                1.0,
            )
        )
        increment_differences.append(
            {
                "output": output_name,
                "medium_vs_fine_relative": _relative_difference(
                    float(medium_increment[output_name]),
                    float(fine_increment[output_name]),
                    scale,
                ),
            }
        )
    increment_max_relative = max(
        float(row["medium_vs_fine_relative"]) for row in increment_differences
    )

    response_rows: list[dict[str, object]] = []
    for index, ratio in enumerate(np.linspace(0.10, 1.40, 18)):
        design = {
            "rise_mm": 10.0,
            "thickness_mm": 1.8,
            "displacement_ratio": float(ratio),
            "imperfection_ratio": 0.01,
        }
        deck, apex_id, target = build_shallow_arch_deck(design)
        result = solve_deck(
            deck,
            apex_id=apex_id,
            target_displacement_mm=target,
            rise_mm=10.0,
            case_dir=raw_dir / f"response_{index:02d}",
            executable=executable,
        )
        response_rows.append(
            {
                "displacement_ratio": float(ratio),
                **{name: float(result[name]) for name in OUTPUT_NAMES},
                "regime_code": int(result["regime_code"]),
            }
        )
        progress(f"[pilot response] {index + 1:02d}/18")
    response_force = np.asarray([float(row["final_force_n"]) for row in response_rows])
    peak_index = int(np.argmax(response_force))
    peak_response = float(response_force[peak_index])
    post_minimum = float(np.min(response_force[peak_index:]))
    force_drop_fraction = (
        (peak_response - post_minimum) / max(abs(peak_response), 1.0)
    )

    criteria = {
        "analytical_correlation": analytical_nrmse <= 0.03 and analytical_max_error <= 0.05,
        "mesh_sensitivity": mesh_max_relative <= 0.05,
        "increment_sensitivity": increment_max_relative <= 0.03,
        "hard_nonlinearity": force_drop_fraction >= 0.30 and peak_index < len(response_force) - 3,
    }
    summary: dict[str, object] = {
        "solver": {
            "executable": executable,
            "version": _solver_version(executable),
            "sha256": _sha256(Path(executable)),
        },
        "analytical_nrmse": analytical_nrmse,
        "analytical_max_normalized_error": analytical_max_error,
        "mesh_max_relative_change_40_to_80": mesh_max_relative,
        "increment_max_relative_change_002_to_001": increment_max_relative,
        "force_drop_fraction": force_drop_fraction,
        "criteria": criteria,
        "all_passed": bool(all(criteria.values())),
    }
    _write_rows(pilot_dir / "analytical_correlation.csv", analytical_rows)
    _write_rows(pilot_dir / "mesh_sensitivity.csv", mesh_rows)
    _write_rows(pilot_dir / "mesh_differences.csv", mesh_differences)
    _write_rows(pilot_dir / "increment_sensitivity.csv", increment_rows)
    _write_rows(pilot_dir / "increment_differences.csv", increment_differences)
    _write_rows(pilot_dir / "response_scan.csv", response_rows)
    (pilot_dir / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_pilot(pilot_dir, analytical_rows, response_rows)
    return summary


def _plot_pilot(
    pilot_dir: Path,
    analytical_rows: Sequence[Mapping[str, object]],
    response_rows: Sequence[Mapping[str, object]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(pilot_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ratio = [float(row["displacement_ratio"]) for row in analytical_rows]
    axes[0].plot(ratio, [float(row["analytical_force_n"]) for row in analytical_rows], "k-", label="analytical")
    axes[0].plot(ratio, [float(row["fea_force_n"]) for row in analytical_rows], "o", label="CalculiX")
    axes[0].set(title="Two-bar solver correlation", xlabel="displacement / rise", ylabel="actuator force [N]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(
        [float(row["displacement_ratio"]) for row in response_rows],
        [float(row["final_force_n"]) for row in response_rows],
        "o-",
        color="#1f77b4",
    )
    axes[1].set(title="B31 shallow-arch nonlinear response", xlabel="displacement / rise", ylabel="actuator force [N]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pilot_dir / "pilot_response.png", dpi=180)
    plt.close(fig)


def load_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_DATA_SEED,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    """Load a complete dataset while preserving pool/holdout isolation."""

    expected = design_points(pool_size, test_size, seed)
    latest = _read_latest_rows(dataset_path.resolve())
    _validate_resume_design(expected, latest)
    missing = [
        str(row["sample_id"])
        for row in expected
        if not _row_succeeded(latest.get(str(row["sample_id"])))
    ]
    if missing:
        raise RuntimeError(
            f"Dataset is incomplete ({len(missing)} missing/failed). "
            f"First ids: {missing[:5]}"
        )

    def arrays(split: str):
        designs = [row for row in expected if row["split"] == split]
        rows = [latest[str(row["sample_id"])] for row in designs]
        X = np.asarray(
            [[float(row[name]) for name in PARAMETER_NAMES] for row in rows],
            dtype=float,
        )
        y = np.asarray(
            [[float(row[name]) for name in OUTPUT_NAMES] for row in rows],
            dtype=float,
        )
        regimes = np.asarray([int(row["regime_code"]) for row in rows], dtype=int)
        ids = [str(row["sample_id"]) for row in rows]
        return X, y, regimes, ids

    X_pool, y_pool, regime_pool, pool_ids = arrays("pool")
    X_test, y_test, regime_test, test_ids = arrays("test")
    return (
        X_pool,
        y_pool,
        regime_pool,
        X_test,
        y_test,
        regime_test,
        pool_ids,
        test_ids,
    )


def farthest_point_order(points_unit: np.ndarray, seed: int) -> np.ndarray:
    """Return a deterministic nested maximin order for a finite pool."""

    points = np.asarray(points_unit, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("points_unit must be a non-empty two-dimensional array.")
    rng = np.random.default_rng(seed)
    next_index = int(rng.integers(len(points)))
    selected = np.zeros(len(points), dtype=bool)
    min_dist = np.full(len(points), np.inf, dtype=float)
    order: list[int] = []
    for _ in range(len(points)):
        order.append(next_index)
        selected[next_index] = True
        distances = np.linalg.norm(points - points[next_index], axis=1)
        min_dist = np.minimum(min_dist, distances)
        min_dist[selected] = -np.inf
        if len(order) < len(points):
            next_index = int(np.argmax(min_dist))
    return np.asarray(order, dtype=int)


def committee_replay_indices(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    *,
    seed: int,
    budget: int,
    n_initial: int = ACTIVE_INITIAL,
    batch_size: int = ACTIVE_BATCH,
    min_dist: float = 0.06,
    acquisition_fn: Callable[..., object] = acquisition_scores,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Select a committee path without exposing unselected pool labels."""

    if not 2 <= n_initial < budget <= len(X_pool):
        raise ValueError("Require 2 <= n_initial < budget <= pool size.")
    if (budget - n_initial) % batch_size:
        raise ValueError("budget - n_initial must be divisible by batch_size.")
    pool_unit = normalize_to_unit(X_pool, BOUNDS)
    maximin_order = farthest_point_order(pool_unit, seed)
    selected = maximin_order[:n_initial].tolist()
    taken = np.zeros(len(X_pool), dtype=bool)
    taken[selected] = True
    trace: list[dict[str, object]] = []
    for round_index in range((budget - n_initial) // batch_size):
        selected_array = np.asarray(selected, dtype=int)
        # This is the leakage barrier: only selected labels are passed.
        result = acquisition_fn(
            "committee",
            X_pool[selected_array],
            y_pool[selected_array],
            X_pool,
            explore_floor=0.3,
            random_state=seed + round_index,
            gp_restarts=1,
        )
        indices = diverse_top_k(
            np.asarray(result.scores, dtype=float),
            pool_unit,
            batch_size,
            taken_mask=taken,
            min_dist=min_dist,
        )
        if len(indices) < batch_size:
            already = set(indices.tolist())
            extras = [
                int(index)
                for index in np.argsort(-np.asarray(result.scores), kind="stable")
                if not taken[int(index)] and int(index) not in already
            ][: batch_size - len(indices)]
            if extras:
                indices = np.concatenate([indices, np.asarray(extras, dtype=int)])
        if len(indices) != batch_size:
            raise RuntimeError("Committee replay could not fill the requested batch.")
        taken[indices] = True
        selected.extend(indices.tolist())
        trace.append(
            {
                "round": round_index + 1,
                "indices": indices.tolist(),
                "scores": np.asarray(result.scores)[indices].tolist(),
                "labels_visible_before_selection": len(selected_array),
            }
        )
    return np.asarray(selected, dtype=int), trace


def _gp_model(input_dim: int, seed: int) -> TransformedTargetRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(input_dim),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        + WhiteKernel(1e-4, (1e-8, 1e-1))
    )
    pipeline = Pipeline(
        [
            ("input_scaler", StandardScaler()),
            (
                "regressor",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=False,
                    n_restarts_optimizer=2,
                    random_state=seed,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )


def prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_true: np.ndarray,
    regime_true: np.ndarray,
) -> dict[str, float]:
    """Return global, tail, transition, and regime metrics."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction and truth arrays must have the same shape.")
    metrics: dict[str, float] = {}
    nrmse_values = []
    r2_values = []
    transition_values = []
    transition_mask = np.asarray(regime_true, dtype=int) == 1
    for index, output_name in enumerate(OUTPUT_NAMES):
        truth = y_true[:, index]
        prediction = y_pred[:, index]
        error = np.abs(truth - prediction)
        rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
        scale = float(np.std(truth))
        nrmse = rmse / scale if scale > 1e-12 else float("inf")
        metrics[f"{output_name}_rmse"] = rmse
        metrics[f"{output_name}_nrmse"] = nrmse
        metrics[f"{output_name}_mae"] = float(mean_absolute_error(truth, prediction))
        metrics[f"{output_name}_r2"] = float(r2_score(truth, prediction))
        metrics[f"{output_name}_p95_abs_error"] = float(np.percentile(error, 95))
        metrics[f"{output_name}_worst_abs_error"] = float(np.max(error))
        nrmse_values.append(nrmse)
        r2_values.append(metrics[f"{output_name}_r2"])
        if np.any(transition_mask) and scale > 1e-12:
            transition_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        truth[transition_mask], prediction[transition_mask]
                    )
                )
                / scale
            )
            transition_values.append(transition_rmse)
    target_displacement = X_true[:, 0] * X_true[:, 2]
    predicted_peak_displacement = y_pred[:, OUTPUT_NAMES.index("pre_peak_displacement_mm")]
    transition_width = 0.10 * X_true[:, 0]
    predicted_regime = np.where(
        target_displacement < predicted_peak_displacement - transition_width,
        0,
        np.where(
            target_displacement <= predicted_peak_displacement + transition_width,
            1,
            2,
        ),
    )
    actual_post = np.asarray(regime_true) == 2
    predicted_post = predicted_regime == 2
    tp = int(np.count_nonzero(actual_post & predicted_post))
    fp = int(np.count_nonzero(~actual_post & predicted_post))
    fn = int(np.count_nonzero(actual_post & ~predicted_post))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics.update(
        {
            "aggregate_nrmse": float(np.mean(nrmse_values)),
            "aggregate_r2": float(np.mean(r2_values)),
            "transition_aggregate_nrmse": (
                float(np.mean(transition_values)) if transition_values else float("nan")
            ),
            "regime_accuracy": float(np.mean(predicted_regime == regime_true)),
            "post_limit_precision": float(precision),
            "post_limit_recall": float(recall),
            "post_limit_f1": float(f1),
        }
    )
    return metrics


def _fit_predict_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    model = _gp_model(X_train.shape[1], seed)
    model.fit(X_train, y_train)
    prediction = np.asarray(model.predict(X_test), dtype=float)
    return prediction, time.perf_counter() - started


def _coverage_metrics(
    selected_unit: np.ndarray,
    test_unit: np.ndarray,
) -> tuple[float, float]:
    distances = np.linalg.norm(
        test_unit[:, None, :] - selected_unit[None, :, :], axis=2
    )
    nearest = np.min(distances, axis=1)
    return float(np.mean(nearest)), float(np.max(nearest))


def paired_statistics(
    sampling_rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20260727,
) -> list[dict[str, object]]:
    """Calculate paired inference with seed as the independent unit."""

    grouped: dict[int, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in sampling_rows:
        grouped[int(row["seed"])][str(row["sampling"])] = row
    budgets = sorted({int(row["budget"]) for row in sampling_rows})
    statistics: list[dict[str, object]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for budget in budgets:
        pairs = []
        for seed in sorted(grouped):
            candidates = [
                row
                for row in sampling_rows
                if int(row["seed"]) == seed and int(row["budget"]) == budget
            ]
            by_sampling = {str(row["sampling"]): row for row in candidates}
            if {"static_maximin", "committee"} <= set(by_sampling):
                pairs.append(
                    (
                        float(by_sampling["static_maximin"]["aggregate_nrmse"]),
                        float(by_sampling["committee"]["aggregate_nrmse"]),
                        float(by_sampling["static_maximin"]["transition_aggregate_nrmse"]),
                        float(by_sampling["committee"]["transition_aggregate_nrmse"]),
                    )
                )
        values = np.asarray(pairs, dtype=float)
        if len(values) < 2:
            raise RuntimeError(f"Budget {budget} has fewer than two paired seeds.")
        static, committee = values[:, 0], values[:, 1]
        paired_improvement = 100.0 * (static - committee) / static
        bootstrap_means = np.mean(
            paired_improvement[
                rng.integers(0, len(paired_improvement), size=(bootstrap_samples, len(paired_improvement)))
            ],
            axis=1,
        )
        difference = static - committee
        difference_std = float(np.std(difference, ddof=1))
        try:
            wilcoxon_p = float(wilcoxon(committee, static).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        transition_static = values[:, 2]
        transition_committee = values[:, 3]
        statistics.append(
            {
                "budget": budget,
                "seeds": len(values),
                "static_nrmse_mean": float(np.mean(static)),
                "static_nrmse_std": float(np.std(static)),
                "committee_nrmse_mean": float(np.mean(committee)),
                "committee_nrmse_std": float(np.std(committee)),
                "ratio_of_means_improvement_pct": float(
                    100.0 * (np.mean(static) - np.mean(committee)) / np.mean(static)
                ),
                "paired_improvement_mean_pct": float(np.mean(paired_improvement)),
                "paired_improvement_median_pct": float(np.median(paired_improvement)),
                "paired_improvement_iqr_pct": float(
                    np.percentile(paired_improvement, 75)
                    - np.percentile(paired_improvement, 25)
                ),
                "paired_bootstrap_ci95_low_pct": float(np.percentile(bootstrap_means, 2.5)),
                "paired_bootstrap_ci95_high_pct": float(np.percentile(bootstrap_means, 97.5)),
                "committee_wins": int(np.count_nonzero(committee < static)),
                "paired_t_pvalue": float(ttest_rel(committee, static).pvalue),
                "wilcoxon_pvalue": wilcoxon_p,
                "cohen_dz": (
                    float(np.mean(difference) / difference_std)
                    if difference_std > 1e-12
                    else 0.0
                ),
                "transition_static_nrmse_mean": float(np.nanmean(transition_static)),
                "transition_committee_nrmse_mean": float(np.nanmean(transition_committee)),
            }
        )
    return statistics


def validation_gates(
    statistic: Mapping[str, object],
    *,
    minimum_mean_improvement_pct: float = MINIMUM_MEAN_IMPROVEMENT_PCT,
    minimum_win_fraction: float = MINIMUM_WIN_FRACTION,
) -> dict[str, bool]:
    """Evaluate the predeclared committee gates for one FEA budget."""

    seeds = int(statistic["seeds"])
    return {
        "mean_improvement": (
            float(statistic["paired_improvement_mean_pct"])
            >= minimum_mean_improvement_pct
        ),
        "positive_bootstrap_ci": (
            float(statistic["paired_bootstrap_ci95_low_pct"]) > 0.0
        ),
        "win_fraction": (
            int(statistic["committee_wins"])
            >= math.ceil(minimum_win_fraction * seeds)
        ),
        "transition_not_worse": (
            float(statistic["transition_committee_nrmse_mean"])
            <= float(statistic["transition_static_nrmse_mean"])
        ),
    }


def summarize_model_quality(
    sampling_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return seed-averaged per-output and classification quality by budget."""

    metric_names = (
        "final_force_n_nrmse",
        "pre_peak_force_n_nrmse",
        "pre_peak_displacement_mm_nrmse",
        "strain_energy_nmm_nrmse",
        "aggregate_r2",
        "regime_accuracy",
    )
    result: dict[str, dict[str, dict[str, float]]] = {}
    for budget in sorted({int(row["budget"]) for row in sampling_rows}):
        result[str(budget)] = {}
        for sampling in ("static_maximin", "committee"):
            rows = [
                row
                for row in sampling_rows
                if int(row["budget"]) == budget
                and str(row["sampling"]) == sampling
            ]
            result[str(budget)][sampling] = {
                name: float(np.mean([float(row[name]) for row in rows]))
                for name in metric_names
            }
    return result


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "scipy", "scikit-learn"):
        try:
            versions[distribution] = package_version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def _git_text(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def write_provenance(
    results_dir: Path,
    dataset_path: Path,
    summary: Mapping[str, object],
    pilot: Mapping[str, object],
) -> dict[str, object]:
    """Write a machine-readable audit trail for the qualification evidence."""

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "commit": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("branch", "--show-current"),
            "working_tree_dirty": bool(_git_text("status", "--porcelain")),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "solver": pilot.get("solver", {}),
        "finite_element_model": {
            "solver": "CalculiX",
            "analysis": "static geometric nonlinearity, displacement control",
            "element": "B31 beam",
            "mesh_elements": DEFAULT_MESH_ELEMENTS,
            "maximum_increment": DEFAULT_MAX_INCREMENT,
            "unit_system": "N-mm-MPa",
            "youngs_modulus_mpa": YOUNGS_MODULUS_MPA,
            "poissons_ratio": POISSONS_RATIO,
            "section_width_mm": SECTION_WIDTH_MM,
            "half_span_mm": HALF_SPAN_MM,
        },
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": _sha256(dataset_path),
            "data_seed": DEFAULT_DATA_SEED,
            "pool_size": summary["dataset"]["pool_size"],
            "holdout_size": summary["dataset"]["test_size"],
        },
        "sampling_protocol": {
            "initial_points": ACTIVE_INITIAL,
            "batch_size": ACTIVE_BATCH,
            "budgets": summary["budgets"],
            "selection_seeds": summary["benchmark_seeds"],
            "primary_budget": summary["primary_budget"],
            "comparator": "nested farthest-point maximin",
            "committee": "Gaussian Process plus Random Forest disagreement",
            "final_model": "same Gaussian Process for both sampling methods",
            "holdout_labels_visible_to_acquisition": False,
        },
        "qualification": {
            "pilot_all_passed": bool(pilot.get("all_passed")),
            "success_criteria": summary["success_criteria"],
            "validation_by_budget": summary["validation_by_budget"],
        },
        "source_sha256": {
            "benchmark": _sha256(Path(__file__)),
            "active_learning": _sha256(
                REPO_ROOT / "pylcss" / "surrogate_modeling" / "active_learning.py"
            ),
        },
    }
    (results_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def run_benchmark(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    data_seed: int = DEFAULT_DATA_SEED,
    benchmark_seeds: int = DEFAULT_BENCHMARK_SEEDS,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run the 20-seed equal-budget maximin/committee comparison."""

    (
        X_pool,
        y_pool,
        regime_pool,
        X_test,
        y_test,
        regime_test,
        pool_ids,
        _,
    ) = load_dataset(
        dataset_path,
        pool_size=pool_size,
        test_size=test_size,
        seed=data_seed,
    )
    budgets = tuple(sorted(set(int(value) for value in budgets)))
    if not budgets or min(budgets) < ACTIVE_INITIAL + ACTIVE_BATCH:
        raise ValueError("Every budget must allow at least one committee batch.")
    if max(budgets) > len(X_pool):
        raise ValueError("A budget cannot exceed the finite pool size.")
    if any((budget - ACTIVE_INITIAL) % ACTIVE_BATCH for budget in budgets):
        raise ValueError("Every budget must equal initial + whole batches.")

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    pool_unit = normalize_to_unit(X_pool, BOUNDS)
    test_unit = normalize_to_unit(X_test, BOUNDS)
    sampling_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    traces: dict[str, object] = {}
    total_fits = benchmark_seeds * len(budgets) * 2
    fit_index = 0
    for seed in range(benchmark_seeds):
        static_order = farthest_point_order(pool_unit, seed)
        committee_order, trace = committee_replay_indices(
            X_pool,
            y_pool,
            seed=seed,
            budget=max(budgets),
        )
        if not np.array_equal(
            static_order[:ACTIVE_INITIAL], committee_order[:ACTIVE_INITIAL]
        ):
            raise RuntimeError("Committee and maximin initial designs diverged.")
        traces[str(seed)] = {
            "initial_sample_ids": [
                pool_ids[index] for index in committee_order[:ACTIVE_INITIAL]
            ],
            "rounds": [
                {
                    **round_trace,
                    "sample_ids": [
                        pool_ids[index] for index in round_trace["indices"]
                    ],
                }
                for round_trace in trace
            ],
        }
        for budget in budgets:
            for sampling, indices in (
                ("static_maximin", static_order[:budget]),
                ("committee", committee_order[:budget]),
            ):
                fit_index += 1
                progress(
                    f"[benchmark {fit_index:03d}/{total_fits:03d}] "
                    f"seed={seed:02d} budget={budget:02d} {sampling}"
                )
                prediction, fit_seconds = _fit_predict_gp(
                    X_pool[indices], y_pool[indices], X_test, seed
                )
                metrics = prediction_metrics(
                    y_test, prediction, X_test, regime_test
                )
                sampling_rows.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "sampling": sampling,
                        "architecture": "Gaussian Process",
                        "fit_seconds": fit_seconds,
                        **metrics,
                    }
                )
                coverage_mean, coverage_max = _coverage_metrics(
                    pool_unit[indices], test_unit
                )
                diagnostics.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "sampling": sampling,
                        "pre_limit_selected": int(np.count_nonzero(regime_pool[indices] == 0)),
                        "transition_selected": int(np.count_nonzero(regime_pool[indices] == 1)),
                        "post_limit_selected": int(np.count_nonzero(regime_pool[indices] == 2)),
                        "coverage_mean": coverage_mean,
                        "coverage_max": coverage_max,
                    }
                )

    stats = paired_statistics(sampling_rows)
    model_quality_by_budget = summarize_model_quality(sampling_rows)
    primary_budget = PRIMARY_BUDGET if PRIMARY_BUDGET in budgets else max(budgets)
    primary = next(row for row in stats if int(row["budget"]) == primary_budget)
    validation_by_budget = {}
    for row in stats:
        gates = validation_gates(row)
        validation_by_budget[str(int(row["budget"]))] = {
            "validated": all(gates.values()),
            "gates": gates,
        }
    committee_validated = bool(
        validation_by_budget[str(primary_budget)]["validated"]
    )
    validated_budgets = [
        int(budget)
        for budget, result in validation_by_budget.items()
        if result["validated"]
    ]
    if committee_validated:
        primary_decision = (
            f"Committee validated at the predeclared {primary_budget}-FEA budget"
        )
    elif float(primary["paired_bootstrap_ci95_high_pct"]) < 0.0:
        primary_decision = (
            f"Static maximin is superior at the predeclared {primary_budget}-FEA budget"
        )
    else:
        primary_decision = (
            f"Committee advantage remains unproven at the predeclared "
            f"{primary_budget}-FEA budget"
        )
    if validated_budgets:
        decision = (
            f"Committee passes all qualification gates at "
            f"{', '.join(map(str, validated_budgets))} FEA in the secondary "
            f"budget-wise analysis; its advantage remains unproven at the "
            f"predeclared primary {primary_budget}-FEA budget"
            if not committee_validated
            else primary_decision
        )
    else:
        decision = primary_decision

    latest = _read_latest_rows(dataset_path)
    elapsed = np.asarray(
        [float(row["elapsed_s"]) for row in latest.values() if _row_succeeded(row)],
        dtype=float,
    )
    dataset_info = {
        "pool_size": len(X_pool),
        "test_size": len(X_test),
        "successful_fea": len(X_pool) + len(X_test),
        "failed_fea": 0,
        "regime_counts_pool": {
            str(code): int(np.count_nonzero(regime_pool == code)) for code in (0, 1, 2)
        },
        "regime_counts_test": {
            str(code): int(np.count_nonzero(regime_test == code)) for code in (0, 1, 2)
        },
        "elapsed_s_mean": float(np.mean(elapsed)),
        "elapsed_s_total": float(np.sum(elapsed)),
        "input_bounds": {
            name: list(bound) for name, bound in zip(PARAMETER_NAMES, BOUNDS)
        },
    }
    summary = {
        "dataset": dataset_info,
        "budgets": list(budgets),
        "benchmark_seeds": benchmark_seeds,
        "primary_budget": primary_budget,
        "analysis_status": {
            "primary_budget_analysis": "confirmatory",
            "other_budget_gate_checks": "secondary",
        },
        "paired_statistics": stats,
        "model_quality_by_budget": model_quality_by_budget,
        "success_criteria": {
            "minimum_mean_improvement_pct": MINIMUM_MEAN_IMPROVEMENT_PCT,
            "bootstrap_ci_lower_must_exceed_zero": True,
            "minimum_win_fraction": MINIMUM_WIN_FRACTION,
            "transition_error_must_not_worsen": True,
        },
        "validation_by_budget": validation_by_budget,
        "validated_budgets": validated_budgets,
        "committee_validated": committee_validated,
        "primary_decision": primary_decision,
        "decision": decision,
        "product_recommendation": (
            f"Keep static maximin with a Gaussian Process as the safe default through "
            f"{primary_budget} FEA. Use committee sampling as an opt-in strategy for "
            "known hard-nonlinear limit-point problems when the available evidence "
            f"supports the chosen budget; in this study the secondary analysis passes "
            f"all four gates at {', '.join(map(str, validated_budgets)) or 'no tested'} "
            "FEA."
        ),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(results_dir / "sampling_runs.csv", sampling_rows)
    _write_rows(results_dir / "sampling_diagnostics.csv", diagnostics)
    _write_rows(results_dir / "paired_statistics.csv", stats)
    (results_dir / "selection_traces.json").write_text(
        json.dumps(traces, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_benchmark(
        results_dir, sampling_rows, stats, traces, X_pool, primary_budget
    )
    pilot_summary_path = results_dir / "pilot" / "pilot_summary.json"
    pilot_summary = (
        json.loads(pilot_summary_path.read_text(encoding="utf-8"))
        if pilot_summary_path.is_file()
        else {}
    )
    (results_dir / "REPORT.md").write_text(
        _markdown_report(summary, stats, dataset_info, pilot_summary),
        encoding="utf-8",
    )
    write_provenance(results_dir, dataset_path, summary, pilot_summary)
    return summary


def replacement_statistics(
    rows: Sequence[Mapping[str, object]],
    reference_budgets: Sequence[int] = REPLACEMENT_REFERENCE_BUDGETS,
    *,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20260728,
) -> list[dict[str, object]]:
    """Compare committee-64 with each maximin reference using paired seeds."""

    output_metrics = tuple(f"{name}_nrmse" for name in OUTPUT_NAMES)
    by_seed_method = {
        (int(row["seed"]), str(row["method"])): row for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    rng = np.random.default_rng(bootstrap_seed)
    comparisons: list[dict[str, object]] = []
    for budget in reference_budgets:
        committee_rows = [by_seed_method[(seed, "committee_64")] for seed in seeds]
        reference_rows = [
            by_seed_method[(seed, f"maximin_{int(budget)}")] for seed in seeds
        ]
        committee_error = np.asarray(
            [float(row["aggregate_nrmse"]) for row in committee_rows]
        )
        reference_error = np.asarray(
            [float(row["aggregate_nrmse"]) for row in reference_rows]
        )
        paired_degradation = 100.0 * (
            committee_error - reference_error
        ) / reference_error
        bootstrap_means = np.mean(
            paired_degradation[
                rng.integers(
                    0,
                    len(paired_degradation),
                    size=(bootstrap_samples, len(paired_degradation)),
                )
            ],
            axis=1,
        )
        committee_quality = {
            metric: float(np.mean([float(row[metric]) for row in committee_rows]))
            for metric in output_metrics
        }
        reference_quality = {
            metric: float(np.mean([float(row[metric]) for row in reference_rows]))
            for metric in output_metrics
        }
        output_degradation = {
            metric: float(
                100.0
                * (committee_quality[metric] - reference_quality[metric])
                / max(reference_quality[metric], 1e-12)
            )
            for metric in output_metrics
        }
        committee_transition = float(
            np.mean(
                [float(row["transition_aggregate_nrmse"]) for row in committee_rows]
            )
        )
        reference_transition = float(
            np.mean(
                [float(row["transition_aggregate_nrmse"]) for row in reference_rows]
            )
        )
        transition_degradation = float(
            100.0
            * (committee_transition - reference_transition)
            / max(reference_transition, 1e-12)
        )
        committee_nrmse_mean = float(np.mean(committee_error))
        committee_r2_mean = float(
            np.mean([float(row["aggregate_r2"]) for row in committee_rows])
        )
        gates = {
            "aggregate_noninferiority": bool(
                np.percentile(bootstrap_means, 95.0)
                <= REPLACEMENT_AGGREGATE_MARGIN_PCT
            ),
            "every_output_within_margin": bool(
                max(output_degradation.values()) <= REPLACEMENT_OUTPUT_MARGIN_PCT
            ),
            "transition_within_margin": bool(
                transition_degradation <= REPLACEMENT_TRANSITION_MARGIN_PCT
            ),
            "absolute_nrmse": bool(
                committee_nrmse_mean <= REPLACEMENT_MAX_NRMSE
            ),
            "absolute_r2": bool(committee_r2_mean >= REPLACEMENT_MIN_R2),
        }
        comparisons.append(
            {
                "reference_budget": int(budget),
                "seeds": len(seeds),
                "committee_64_nrmse_mean": committee_nrmse_mean,
                "committee_64_nrmse_std": float(np.std(committee_error)),
                "maximin_reference_nrmse_mean": float(np.mean(reference_error)),
                "maximin_reference_nrmse_std": float(np.std(reference_error)),
                "paired_degradation_mean_pct": float(np.mean(paired_degradation)),
                "paired_degradation_ci95_low_pct": float(
                    np.percentile(bootstrap_means, 2.5)
                ),
                "paired_degradation_ci95_high_pct": float(
                    np.percentile(bootstrap_means, 97.5)
                ),
                "paired_degradation_one_sided_upper95_pct": float(
                    np.percentile(bootstrap_means, 95.0)
                ),
                "committee_better_or_equal_seeds": int(
                    np.count_nonzero(committee_error <= reference_error)
                ),
                "committee_64_aggregate_r2_mean": committee_r2_mean,
                "committee_64_transition_nrmse_mean": committee_transition,
                "reference_transition_nrmse_mean": reference_transition,
                "transition_degradation_pct": transition_degradation,
                **{
                    f"committee_64_{metric}": committee_quality[metric]
                    for metric in output_metrics
                },
                **{
                    f"reference_{metric}": reference_quality[metric]
                    for metric in output_metrics
                },
                **{
                    f"degradation_{metric}_pct": output_degradation[metric]
                    for metric in output_metrics
                },
                "gates": gates,
                "replacement_passed": all(gates.values()),
            }
        )
    return comparisons


def run_replacement_test(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    data_seed: int = DEFAULT_DATA_SEED,
    benchmark_seeds: int = DEFAULT_BENCHMARK_SEEDS,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Test whether committee-64 can replace a strong maximin-100 design."""

    (
        X_pool,
        y_pool,
        regime_pool,
        X_test,
        y_test,
        regime_test,
        pool_ids,
        _,
    ) = load_dataset(
        dataset_path,
        pool_size=pool_size,
        test_size=test_size,
        seed=data_seed,
    )
    if max(REPLACEMENT_REFERENCE_BUDGETS) > len(X_pool):
        raise ValueError("Replacement reference budget exceeds the finite pool.")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rows: list[dict[str, object]] = []
    traces: dict[str, object] = {}
    fits_per_seed = 1 + len(REPLACEMENT_REFERENCE_BUDGETS)
    fit_index = 0
    for seed in range(benchmark_seeds):
        static_order = farthest_point_order(normalize_to_unit(X_pool, BOUNDS), seed)
        committee_order, committee_trace = committee_replay_indices(
            X_pool,
            y_pool,
            seed=seed,
            budget=REPLACEMENT_ACTIVE_BUDGET,
        )
        selections = [("committee_64", committee_order)] + [
            (f"maximin_{budget}", static_order[:budget])
            for budget in REPLACEMENT_REFERENCE_BUDGETS
        ]
        traces[str(seed)] = {
            "committee_64_sample_ids": [pool_ids[index] for index in committee_order],
            "maximin_100_sample_ids": [
                pool_ids[index]
                for index in static_order[: max(REPLACEMENT_REFERENCE_BUDGETS)]
            ],
            "committee_rounds": committee_trace,
        }
        for method, indices in selections:
            fit_index += 1
            progress(
                f"[replacement {fit_index:03d}/{benchmark_seeds * fits_per_seed:03d}] "
                f"seed={seed:02d} {method}"
            )
            prediction, fit_seconds = _fit_predict_gp(
                X_pool[indices], y_pool[indices], X_test, seed
            )
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "fea_labels": len(indices),
                    "architecture": "Gaussian Process",
                    "fit_seconds": fit_seconds,
                    **prediction_metrics(y_test, prediction, X_test, regime_test),
                }
            )

    comparisons = replacement_statistics(rows)
    primary = next(
        row
        for row in comparisons
        if int(row["reference_budget"]) == max(REPLACEMENT_REFERENCE_BUDGETS)
    )
    passing_budgets = [
        int(row["reference_budget"])
        for row in comparisons
        if bool(row["replacement_passed"])
    ]
    equivalent_budget = max(passing_budgets) if passing_budgets else None
    validated_reduction_pct = (
        float(
            100.0
            * (equivalent_budget - REPLACEMENT_ACTIVE_BUDGET)
            / equivalent_budget
        )
        if equivalent_budget
        else None
    )
    replacement_validated = bool(primary["replacement_passed"])
    if replacement_validated:
        decision = "64 committee-selected FEA runs can replace the 100-FEA maximin reference under the frozen margins"
    else:
        decision = "64 committee-selected FEA runs cannot be claimed equivalent to the 100-FEA maximin reference under the frozen margins"
    summary = {
        "question": "Can 64 committee-selected FEA labels replace 100 maximin-selected FEA labels?",
        "analysis_status": "confirmatory non-inferiority test",
        "dataset_sha256": _sha256(dataset_path),
        "pool_size": len(X_pool),
        "holdout_size": len(X_test),
        "benchmark_seeds": benchmark_seeds,
        "active_budget": REPLACEMENT_ACTIVE_BUDGET,
        "reference_budgets": list(REPLACEMENT_REFERENCE_BUDGETS),
        "frozen_acceptance_criteria": {
            "aggregate_one_sided_upper95_degradation_pct_max": REPLACEMENT_AGGREGATE_MARGIN_PCT,
            "each_output_mean_degradation_pct_max": REPLACEMENT_OUTPUT_MARGIN_PCT,
            "transition_mean_degradation_pct_max": REPLACEMENT_TRANSITION_MARGIN_PCT,
            "committee_aggregate_nrmse_max": REPLACEMENT_MAX_NRMSE,
            "committee_aggregate_r2_min": REPLACEMENT_MIN_R2,
        },
        "comparisons": comparisons,
        "passing_reference_budgets": passing_budgets,
        "equivalent_reference_budget": equivalent_budget,
        "validated_fea_reduction_pct_vs_equivalent_budget": validated_reduction_pct,
        "replacement_validated": replacement_validated,
        "nominal_fea_reduction_pct_if_validated": 36.0,
        "decision": decision,
    }
    replacement_dir = results_dir / "replacement_test"
    replacement_dir.mkdir(parents=True, exist_ok=True)
    flat_comparisons = [
        {
            **{key: value for key, value in row.items() if key != "gates"},
            **{f"gate_{key}": value for key, value in row["gates"].items()},
        }
        for row in comparisons
    ]
    _write_rows(replacement_dir / "replacement_runs.csv", rows)
    _write_rows(replacement_dir / "replacement_statistics.csv", flat_comparisons)
    (replacement_dir / "selection_traces.json").write_text(
        json.dumps(traces, indent=2) + "\n", encoding="utf-8"
    )
    (replacement_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_replacement_test(replacement_dir, comparisons)
    (replacement_dir / "REPORT.md").write_text(
        _replacement_markdown_report(summary), encoding="utf-8"
    )
    pilot_path = results_dir / "pilot" / "pilot_summary.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _git_text("rev-parse", "HEAD"),
        "repository_branch": _git_text("branch", "--show-current"),
        "working_tree_dirty": bool(_git_text("status", "--porcelain")),
        "runtime": {
            "python": sys.version.split()[0],
            "packages": _package_versions(),
        },
        "solver": pilot.get("solver", {}),
        "dataset_sha256": _sha256(dataset_path),
        "benchmark_source_sha256": _sha256(Path(__file__)),
        "protocol": {
            "same_untouched_holdout": True,
            "same_final_model": "Gaussian Process",
            "reference_sampling": "nested farthest-point maximin",
            "active_sampling": "GP-RF committee",
            "selection_seeds": benchmark_seeds,
            "criteria": summary["frozen_acceptance_criteria"],
        },
    }
    (replacement_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _plot_replacement_test(
    replacement_dir: Path,
    comparisons: Sequence[Mapping[str, object]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(replacement_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = [int(row["reference_budget"]) for row in comparisons]
    reference = [float(row["maximin_reference_nrmse_mean"]) for row in comparisons]
    reference_std = [float(row["maximin_reference_nrmse_std"]) for row in comparisons]
    committee = float(comparisons[0]["committee_64_nrmse_mean"])
    committee_std = float(comparisons[0]["committee_64_nrmse_std"])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.errorbar(
        budgets,
        reference,
        yerr=reference_std,
        marker="o",
        capsize=3,
        color="#444444",
        label="maximin reference",
    )
    ax.axhline(committee, color="#d62728", label="committee, 64 FEA")
    ax.fill_between(
        budgets,
        committee - committee_std,
        committee + committee_std,
        color="#d62728",
        alpha=0.12,
    )
    ax.set(
        xlabel="maximin reference FEA labels",
        ylabel="holdout aggregate NRMSE",
        title="Can committee-64 replace maximin-100?",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(replacement_dir / "replacement_curve.png", dpi=180)
    plt.close(fig)


def _replacement_markdown_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# 64-versus-100 FEA replacement qualification",
        "",
        "## Question and protocol",
        "",
        "This confirmatory test asks whether a Gaussian Process trained with 64 GP-RF-committee-selected real FEA labels can replace the same Gaussian Process trained with 100 strong nested-maximin real FEA labels. Both methods use the same 160-case real CalculiX pool, the same untouched 64-case real FEA holdout, and 20 paired selection seeds.",
        "",
        "The comparison does not claim that a surrogate is identical to an executed FEA result. It tests predictive accuracy on designs whose real CalculiX labels were never exposed during training or acquisition.",
        "",
        "## Frozen acceptance criteria",
        "",
        "Before running the 100-label comparison, replacement required all of the following: one-sided 95% upper bound on aggregate degradation <=10%; every output's mean NRMSE degradation <=15%; transition-band degradation <=10%; committee aggregate NRMSE <=0.05; and committee aggregate R2 >=0.99.",
        "",
        "## Budget map",
        "",
        "| Maximin reference budget | Committee-64 NRMSE | Reference NRMSE | Mean degradation | One-sided 95% upper | Transition degradation | Seeds no worse | All gates |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["comparisons"]:
        lines.append(
            f"| {int(row['reference_budget'])} | "
            f"{float(row['committee_64_nrmse_mean']):.4f} | "
            f"{float(row['maximin_reference_nrmse_mean']):.4f} | "
            f"{float(row['paired_degradation_mean_pct']):+.2f}% | "
            f"{float(row['paired_degradation_one_sided_upper95_pct']):+.2f}% | "
            f"{float(row['transition_degradation_pct']):+.2f}% | "
            f"{int(row['committee_better_or_equal_seeds'])}/{int(row['seeds'])} | "
            f"{'PASS' if row['replacement_passed'] else 'FAIL'} |"
        )
    primary = next(
        row
        for row in summary["comparisons"]
        if int(row["reference_budget"]) == 100
    )
    lines.extend(
        [
            "",
            "## Primary 64-versus-100 decision",
            "",
            f"**{summary['decision']}.**",
            "",
            f"At 100 reference labels, committee-64 has mean aggregate NRMSE `{float(primary['committee_64_nrmse_mean']):.4f}` versus `{float(primary['maximin_reference_nrmse_mean']):.4f}`. Its mean paired degradation is `{float(primary['paired_degradation_mean_pct']):+.2f}%`, with a one-sided 95% upper bound of `{float(primary['paired_degradation_one_sided_upper95_pct']):+.2f}%`.",
            "",
            f"The largest maximin reference budget passing every frozen gate is `{summary['equivalent_reference_budget']}` FEA. This supports a `{float(summary['validated_fea_reduction_pct_vs_equivalent_budget']):.1f}%` reduction from that tested equivalent budget to 64. The nominal 36% reduction from 100 to 64 is rejected because the primary comparison did not pass.",
            "",
            "## Scope",
            "",
            "The decision applies to this low-dimensional CalculiX shallow-arch snap-through benchmark and its four scalar outputs. It does not automatically transfer to contact, fracture, crash, topology changes, or field surrogates.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_benchmark(
    results_dir: Path,
    sampling_rows: Sequence[Mapping[str, object]],
    stats: Sequence[Mapping[str, object]],
    traces: Mapping[str, object],
    X_pool: np.ndarray,
    primary_budget: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(results_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for sampling, color in (("static_maximin", "#444444"), ("committee", "#d62728")):
        means, stds, budgets = [], [], []
        for budget in sorted({int(row["budget"]) for row in sampling_rows}):
            values = np.asarray(
                [
                    float(row["aggregate_nrmse"])
                    for row in sampling_rows
                    if row["sampling"] == sampling and int(row["budget"]) == budget
                ]
            )
            budgets.append(budget)
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
        ax.errorbar(budgets, means, yerr=stds, marker="o", capsize=3, label=sampling, color=color)
    ax.set(xlabel="successful real FEA labels", ylabel="holdout aggregate NRMSE", title="Hard-nonlinear active-learning comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "learning_curve.png", dpi=180)
    plt.close(fig)

    primary_rows = [row for row in sampling_rows if int(row["budget"]) == primary_budget]
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in primary_rows:
        by_seed[int(row["seed"])][str(row["sampling"])] = float(row["aggregate_nrmse"])
    improvements = [
        100.0 * (values["static_maximin"] - values["committee"]) / values["static_maximin"]
        for _, values in sorted(by_seed.items())
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = ["#2ca02c" if value > 0.0 else "#d62728" for value in improvements]
    ax.bar(np.arange(len(improvements)), improvements, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set(xlabel="independent seed", ylabel="committee improvement [%]", title=f"Paired result at {primary_budget} FEA")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "paired_improvements.png", dpi=180)
    plt.close(fig)

    seed_zero = traces.get("0", {})
    committee_ids = []
    id_to_index = {f"pool_{index:03d}": index for index in range(len(X_pool))}
    for sample_id in seed_zero.get("initial_sample_ids", []):
        committee_ids.append(id_to_index[str(sample_id)])
    for round_trace in seed_zero.get("rounds", []):
        for sample_id in round_trace.get("sample_ids", []):
            committee_ids.append(id_to_index[str(sample_id)])
    committee_indices = np.asarray(committee_ids[:primary_budget], dtype=int)
    static_indices = farthest_point_order(normalize_to_unit(X_pool, BOUNDS), 0)[:primary_budget]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    for ax, name, indices, color in (
        (axes[0], "static maximin", static_indices, "#444444"),
        (axes[1], "committee", committee_indices, "#d62728"),
    ):
        ax.scatter(X_pool[:, 2], X_pool[:, 0], s=10, color="#cccccc", label="pool")
        ax.scatter(X_pool[indices, 2], X_pool[indices, 0], s=28, color=color, label="selected")
        ax.set(title=f"seed 0: {name}", xlabel="displacement / rise")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("arch rise [mm]")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(results_dir / "seed0_selection.png", dpi=180)
    plt.close(fig)


def _markdown_report(
    summary: Mapping[str, object],
    stats: Sequence[Mapping[str, object]],
    dataset_info: Mapping[str, object],
    pilot: Mapping[str, object],
) -> str:
    lines = [
        "# Hard nonlinear CalculiX FEA active-learning qualification",
        "",
        "## Engineering case",
        "",
        "A pin-supported, imperfect shallow arch is discretized with CalculiX B31 beam elements and driven through its first limit point under displacement control. The response contains a peak-force transition and negative tangent stiffness; no artificial response cap is used. Inputs are rise, section thickness, imposed displacement/rise ratio, and geometric imperfection. Outputs are final actuator force, pre-limit peak force, FEA-estimated peak displacement, and signed strain energy.",
        "",
        "## Numerical qualification",
        "",
        f"- Two-bar analytical correlation NRMSE: `{float(pilot.get('analytical_nrmse', float('nan'))):.4%}`.",
        f"- Maximum 40-to-80 element output change: `{float(pilot.get('mesh_max_relative_change_40_to_80', float('nan'))):.4%}`.",
        f"- Maximum 0.02-to-0.01 increment output change: `{float(pilot.get('increment_max_relative_change_002_to_001', float('nan'))):.4%}`.",
        f"- Post-limit force-drop fraction in the pilot scan: `{float(pilot.get('force_drop_fraction', float('nan'))):.2%}`.",
        f"- Pilot acceptance: `{'PASS' if pilot.get('all_passed') else 'FAIL'}`.",
        "",
        "## Dataset and protocol",
        "",
        f"- `{dataset_info['pool_size']}` real FEA pool cases and `{dataset_info['test_size']}` untouched real FEA holdout cases.",
        f"- `{dataset_info['successful_fea']}` successful and `{dataset_info['failed_fea']}` failed FEA solves.",
        f"- Mean/total solver time: `{float(dataset_info['elapsed_s_mean']):.3f} s` / `{float(dataset_info['elapsed_s_total']):.1f} s`.",
        f"- `{summary['benchmark_seeds']}` independent selection seeds. Seed, not model architecture, is the inferential unit.",
        "- Committee and maximin use the same first 12 maximin points, the same finite pool, equal budgets, the same final Gaussian Process, and the same untouched holdout.",
        "- During committee replay only labels belonging to previously selected indices are visible.",
        "",
        "## Paired holdout results",
        "",
        "| FEA budget | Maximin NRMSE | Committee NRMSE | Mean paired improvement | 95% paired bootstrap CI | Wins | paired t p | Wilcoxon p | Strict gates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in stats:
        lines.append(
            f"| {int(row['budget'])} | {float(row['static_nrmse_mean']):.4f} +/- {float(row['static_nrmse_std']):.4f} | "
            f"{float(row['committee_nrmse_mean']):.4f} +/- {float(row['committee_nrmse_std']):.4f} | "
            f"{float(row['paired_improvement_mean_pct']):+.2f}% | "
            f"[{float(row['paired_bootstrap_ci95_low_pct']):+.2f}%, {float(row['paired_bootstrap_ci95_high_pct']):+.2f}%] | "
            f"{int(row['committee_wins'])}/{int(row['seeds'])} | "
            f"{float(row['paired_t_pvalue']):.4g} | {float(row['wilcoxon_pvalue']):.4g} | "
            f"{'PASS' if summary['validation_by_budget'][str(int(row['budget']))]['validated'] else 'FAIL'} |"
        )
    selected_quality_budgets = [int(summary["primary_budget"])]
    selected_quality_budgets.extend(
        budget
        for budget in summary["validated_budgets"]
        if budget not in selected_quality_budgets
    )
    lines.extend(
        [
            "",
            "## Per-output model quality",
            "",
            "Seed-averaged normalized errors are shown for the predeclared product budget and every secondary budget that passes all four gates.",
            "",
            "| Budget | Sampling | Final force NRMSE | Peak force NRMSE | Peak displacement NRMSE | Energy NRMSE | Aggregate R2 | Regime accuracy |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for budget in selected_quality_budgets:
        for sampling in ("static_maximin", "committee"):
            quality = summary["model_quality_by_budget"][str(budget)][sampling]
            lines.append(
                f"| {budget} | {sampling} | "
                f"{float(quality['final_force_n_nrmse']):.4f} | "
                f"{float(quality['pre_peak_force_n_nrmse']):.4f} | "
                f"{float(quality['pre_peak_displacement_mm_nrmse']):.4f} | "
                f"{float(quality['strain_energy_nmm_nrmse']):.4f} | "
                f"{float(quality['aggregate_r2']):.4f} | "
                f"{float(quality['regime_accuracy']):.2%} |"
            )
    lines.extend(
        [
            "",
            "## Qualification decision",
            "",
            f"**{summary['decision']}.**",
            "",
            f"Primary decision: **{summary['primary_decision']}**. The 32-FEA primary budget and its four success gates were fixed before interpreting the results: mean paired improvement of at least 5%, a strictly positive paired-bootstrap confidence interval, at least 60% seed wins, and no transition-band degradation.",
            "",
            "Applying the same gates to the other budgets is a secondary budget-wise analysis, not a replacement for the failed primary decision. At 48 FEA the global improvement is strong and repeatable, but the transition gate misses by a small amount. At 64 FEA all four gates pass. The transition subset contains only five holdout cases, so its budget-to-budget result should be treated as higher-variance evidence.",
            "",
            "## Product recommendation",
            "",
            str(summary["product_recommendation"]),
            "",
            "A dedicated confirmatory replacement test subsequently mapped committee-64 against maximin budgets through 100. Committee-64 passed every frozen non-inferiority gate through maximin-80, supporting a 20% reduction from that tested equivalent budget, but it failed against maximin-100; therefore a 36% reduction claim is rejected. See `replacement_test/REPORT.md`.",
            "",
            "## Reproducibility and traceability",
            "",
            f"- CalculiX version: `{pilot.get('solver', {}).get('version', 'unknown')}`.",
            f"- Solver SHA-256: `{pilot.get('solver', {}).get('sha256', 'unknown')}`.",
            "- Units: `N-mm-MPa`; element: `B31`; default mesh: `40`; maximum increment: `0.02`.",
            "- Full machine-readable environment, source hashes, design seeds, solver identity, and acceptance gates are recorded in `provenance.json`.",
            "",
            "## Interpretation limits",
            "",
            "This is a real CalculiX geometric snap-through benchmark with solver, mesh, and increment qualification. It supports a product decision for low-dimensional scalar responses with a localized limit-point transition. It does not establish universality for contact, fracture, topology changes, or high-dimensional field surrogates.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("pilot", "generate", "benchmark", "replacement", "all"),
        default="all",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--benchmark-seeds", type=int, default=DEFAULT_BENCHMARK_SEEDS)
    parser.add_argument("--budgets", type=_parse_int_list, default=DEFAULT_BUDGETS)
    args = parser.parse_args()

    if args.mode in {"pilot", "all"}:
        pilot = run_pilot(args.results_dir)
        print("pilot:", json.dumps(pilot, indent=2))
        if not pilot["all_passed"]:
            print("Pilot qualification failed; dataset generation was not started.")
            return 2
    if args.mode in {"generate", "all"}:
        status = generate_dataset(
            args.dataset,
            args.results_dir,
            pool_size=args.pool_size,
            test_size=args.test_size,
            seed=args.data_seed,
        )
        print("dataset:", json.dumps(status, indent=2))
        if status["pending"]:
            return 3
    if args.mode in {"benchmark", "all"}:
        summary = run_benchmark(
            args.dataset,
            args.results_dir,
            pool_size=args.pool_size,
            test_size=args.test_size,
            data_seed=args.data_seed,
            benchmark_seeds=args.benchmark_seeds,
            budgets=args.budgets,
        )
        print(json.dumps(summary, indent=2))
        print(f"report: {(args.results_dir / 'REPORT.md').resolve()}")
    if args.mode in {"replacement", "all"}:
        replacement = run_replacement_test(
            args.dataset,
            args.results_dir,
            pool_size=args.pool_size,
            test_size=args.test_size,
            data_seed=args.data_seed,
            benchmark_seeds=args.benchmark_seeds,
        )
        print(json.dumps(replacement, indent=2))
        print(
            "replacement report: "
            f"{(args.results_dir / 'replacement_test' / 'REPORT.md').resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
