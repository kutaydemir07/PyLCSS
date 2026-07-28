# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""OpenRadioss time-history conversion and canonical channel parsing.

OpenRadioss writes its global energy, momentum, mass, time-step, and requested
rigid-wall channels to a binary T-file.  The official ``th_to_csv`` utility is
the supported way to decode that file.  This module deliberately avoids
reverse-engineering the binary format and instead normalises the converter's
CSV columns into a stable PyLCSS contract.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from pylcss.solver_backends.common import resolve_executable


_GLOBAL_CHANNEL_ALIASES = {
    "time_ms": ("TIME",),
    "internal_energy_kj": ("IE", "INTERNALENERGY"),
    "kinetic_energy_kj": ("KE", "KINETICENERGY"),
    "total_energy_kj": ("TE", "TOTALENERGY"),
    "translational_total_energy_kj": (
        "TTE",
        "TRANSLATIONALTOTALENERGY",
    ),
    "delta_total_energy_kj": ("DTE", "DELTATOTALENERGY"),
    "delta_total_energy_relative": (
        "DTEREL",
        "DELTATOTALENERGYRELATIVE",
    ),
    "rotational_kinetic_energy_kj": (
        "RKE",
        "ROTATIONALKINETICENERGY",
    ),
    "contact_energy_kj": ("CE", "CONTACTENERGY"),
    "contact_elastic_energy_kj": ("CEELAST", "ELASTICCONTACTENERGY"),
    "contact_friction_energy_kj": ("CEFRIC", "FRICTIONALCONTACTENERGY"),
    "contact_damping_energy_kj": ("CEDAMP", "DAMPINGCONTACTENERGY"),
    "hourglass_energy_kj": ("HE", "HOURGLASSENERGY"),
    "external_work_kj": ("EFW", "EXTERNALWORK"),
    "mass_tonne": ("MASS", "GLOBALMASS"),
    "timestep_ms": ("DT", "TIMESTEP"),
    "momentum_x": ("XMOM", "XMOMENTUM"),
    "momentum_y": ("YMOM", "YMOMENTUM"),
    "momentum_z": ("ZMOM", "ZMOMENTUM"),
    "global_velocity_x": ("VX", "GLOBALVELOCITYX"),
    "global_velocity_y": ("VY", "GLOBALVELOCITYY"),
    "global_velocity_z": ("VZ", "GLOBALVELOCITYZ"),
}

_WALL_CHANNEL_ALIASES = {
    "rigid_wall_impulse_x_raw": ("FNX", "NORMALFORCEX"),
    "rigid_wall_impulse_y_raw": ("FNY", "NORMALFORCEY"),
    "rigid_wall_impulse_z_raw": ("FNZ", "NORMALFORCEZ"),
    "rigid_wall_tangent_impulse_x_raw": ("FTX", "TANGENTFORCEX"),
    "rigid_wall_tangent_impulse_y_raw": ("FTY", "TANGENTFORCEY"),
    "rigid_wall_tangent_impulse_z_raw": ("FTZ", "TANGENTFORCEZ"),
}


def _normalise_header(value: object) -> str:
    """Return an uppercase alphanumeric representation for fuzzy matching."""
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _finite_float(value: object) -> float:
    try:
        out = float(str(value).strip().replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def find_time_history_file(work_dir: str | Path, job_name: str) -> Optional[Path]:
    """Find the primary T-file produced by a run, newest candidate first."""
    root = Path(work_dir)
    primary = root / f"{job_name}T01"
    if primary.is_file():
        return primary
    patterns = (
        f"{job_name}T01*",
        f"{job_name}_0001.thy",
        f"{job_name}_0001*.thy",
    )
    candidates = []
    for pattern in patterns:
        candidates.extend(
            p
            for p in root.glob(pattern)
            if p.is_file()
            and p.suffix.lower() != ".csv"
            and not p.name.upper().endswith("_TITLES")
        )
    if not candidates:
        return None
    return max(set(candidates), key=lambda p: p.stat().st_mtime)


def resolve_th_to_csv(
    explicit: Optional[str] = None,
    solver_executable: Optional[str] = None,
) -> Optional[str]:
    """Resolve the official T-file converter, including solver siblings."""
    resolved = resolve_executable(
        explicit,
        env_vars=("PYLCSS_OPENRADIOSS_TH2CSV", "OPENRADIOSS_TH2CSV"),
        candidates=(
            "th_to_csv_win64.exe",
            "th_to_csv.exe",
            "th_to_csv_linux64_gf",
            "th_to_csv_linux64",
            "th_to_csv",
        ),
    )
    if resolved:
        return resolved

    if solver_executable:
        solver_path = Path(solver_executable).expanduser()
        search_dirs = [solver_path.parent, solver_path.parent.parent / "exec"]
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for pattern in ("th_to_csv*.exe", "th_to_csv*"):
                for candidate in sorted(directory.glob(pattern)):
                    if candidate.is_file():
                        return str(candidate)
    return None


def convert_time_history_to_csv(
    time_history_file: str | Path,
    converter: str,
    timeout_s: float = 120.0,
) -> Path:
    """Run the official converter and return the generated CSV path."""
    source = Path(time_history_file).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"OpenRadioss time-history file not found: {source}")

    expected = Path(str(source) + ".csv")
    try:
        if (
            expected.is_file()
            and expected.stat().st_mtime
            >= source.stat().st_mtime - 2.0
        ):
            return expected
    except OSError:
        pass
    before = {
        p.resolve()
        for p in source.parent.glob("*.csv")
        if p.is_file()
    }
    env = os.environ.copy()
    converter_path = Path(converter).resolve()
    env["PATH"] = str(converter_path.parent) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        [str(converter_path), str(source)],
        cwd=str(source.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(float(timeout_s), 1.0),
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-2000:]
        raise RuntimeError(
            f"th_to_csv failed with exit code {proc.returncode}:\n{tail}"
        )
    if expected.is_file():
        return expected

    after = [
        p.resolve()
        for p in source.parent.glob("*.csv")
        if p.is_file() and p.resolve() not in before
    ]
    if after:
        return max(after, key=lambda p: p.stat().st_mtime)
    raise RuntimeError(
        "th_to_csv reported success but produced no CSV file next to "
        f"{source.name}."
    )


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a converter CSV while tolerating delimiter and preamble variants."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [], []

    header_index = 0
    delimiter = ","
    for idx, line in enumerate(lines):
        comma = line.count(",")
        semicolon = line.count(";")
        tab = line.count("\t")
        candidate_delimiter, count = max(
            ((",", comma), (";", semicolon), ("\t", tab)),
            key=lambda pair: pair[1],
        )
        upper = line.upper()
        if count > 0 and ("TIME" in upper or idx == 0):
            header_index = idx
            delimiter = candidate_delimiter
            if "TIME" in upper:
                break

    reader = csv.reader(lines[header_index:], delimiter=delimiter)
    parsed = list(reader)
    if not parsed:
        return [], []
    width = len(parsed[0])
    rows = [row[:width] + [""] * max(0, width - len(row)) for row in parsed[1:]]
    return [str(v).strip() for v in parsed[0]], rows


def _match_column(normalized_header: str, aliases: Iterable[str]) -> bool:
    """Match a variable name without confusing global and object channels."""
    for alias in aliases:
        if normalized_header == alias:
            return True
        if len(alias) > 2 and alias in normalized_header:
            return True
        if normalized_header.startswith(alias) and (
            len(normalized_header) == len(alias)
            or not normalized_header[len(alias)].isdigit()
        ):
            return True
    return False


def parse_time_history_csv(path: str | Path) -> Dict[str, object]:
    """Parse an official ``th_to_csv`` result into canonical numeric arrays."""
    csv_path = Path(path)
    headers, rows = _read_csv_rows(csv_path)
    if not headers:
        return {
            "source_csv": str(csv_path),
            "raw_columns": {},
            "warnings": ["OpenRadioss time-history CSV was empty."],
        }

    columns: Dict[str, list[float]] = {header: [] for header in headers}
    for row in rows:
        for idx, header in enumerate(headers):
            columns[header].append(_finite_float(row[idx] if idx < len(row) else ""))

    normalized = {header: _normalise_header(header) for header in headers}
    canonical: Dict[str, object] = {
        "source_csv": str(csv_path.resolve()),
        "raw_columns": {
            header: values for header, values in columns.items()
        },
        "column_map": {},
        "warnings": [],
    }

    for output_name, aliases in _GLOBAL_CHANNEL_ALIASES.items():
        matches = [
            header
            for header in headers
            if _match_column(normalized[header], aliases)
        ]
        if not matches:
            continue
        # Object channels may repeat for several walls. The generated PyLCSS
        # deck contains one wall; prefer the first finite column deterministically.
        selected = next(
            (
                header
                for header in matches
                if np.isfinite(np.asarray(columns[header], dtype=float)).any()
            ),
            matches[0],
        )
        canonical[output_name] = columns[selected]
        canonical["column_map"][output_name] = selected

    for output_name, aliases in _WALL_CHANNEL_ALIASES.items():
        matches = [
            header
            for header in headers
            if (
                "RWALL" in normalized[header]
                or "RIGIDWALL" in normalized[header]
            )
            and _match_column(normalized[header], aliases)
        ]
        if not matches:
            continue
        selected = next(
            (
                header
                for header in matches
                if np.isfinite(
                    np.asarray(columns[header], dtype=float)
                ).any()
            ),
            matches[0],
        )
        canonical[output_name] = columns[selected]
        canonical["column_map"][output_name] = selected

    time_values = np.asarray(canonical.get("time_ms", []), dtype=float)
    finite_time = np.isfinite(time_values)
    if time_values.size and not finite_time.all():
        for key, value in list(canonical.items()):
            if isinstance(value, list) and len(value) == time_values.size:
                canonical[key] = np.asarray(value, dtype=float)[finite_time].tolist()
    if not canonical.get("time_ms"):
        canonical["warnings"].append(
            "No TIME column was found in the OpenRadioss history CSV."
        )
    return canonical


def read_openradioss_time_history(
    work_dir: str | Path,
    job_name: str,
    solver_executable: Optional[str] = None,
    converter: Optional[str] = None,
    timeout_s: float = 120.0,
) -> Dict[str, object]:
    """Locate, convert, and parse the run's primary time-history file."""
    source = find_time_history_file(work_dir, job_name)
    if source is None:
        return {
            "warnings": [
                "OpenRadioss produced no T01/.thy time-history file."
            ]
        }

    converter_path = resolve_th_to_csv(converter, solver_executable)
    if not converter_path:
        return {
            "source_file": str(source),
            "warnings": [
                "OpenRadioss time-history exists but th_to_csv was not found. "
                "Install the official converter or set "
                "PYLCSS_OPENRADIOSS_TH2CSV."
            ],
        }
    try:
        csv_path = convert_time_history_to_csv(
            source, converter_path, timeout_s=timeout_s
        )
        result = parse_time_history_csv(csv_path)
    except Exception as exc:
        return {
            "source_file": str(source),
            "converter": converter_path,
            "warnings": [f"OpenRadioss time-history conversion failed: {exc}"],
        }
    result["source_file"] = str(source.resolve())
    result["converter"] = converter_path
    return result
