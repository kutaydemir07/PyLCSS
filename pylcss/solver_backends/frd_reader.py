# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""CalculiX ``.frd`` result file reader.

CCX writes ASCII FRD by default; this parser handles the standard blocks PyLCSS
needs for visualization: nodal coordinates (block 2C), elements (block 3C), and
result groups (block ``-4`` with components in ``-5`` and per-node data in
``-1`` records terminated by ``-3``).

The engineering result subset includes displacement, stress, reaction/nodal
force (``FORC``), and strain-energy density (``ENER``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np


class FrdStep(TypedDict):
    """Metadata for one result block encountered in an FRD file."""

    step: int
    name: str
    time: float


class FrdResult(TypedDict):
    """Normalized CalculiX result fields returned by :func:`read_frd`."""

    nodes: np.ndarray
    node_ids: np.ndarray
    displacement: np.ndarray
    stress: np.ndarray
    von_mises: np.ndarray
    nodal_force: np.ndarray
    ener: np.ndarray
    steps: list[FrdStep]


def _to_float(token: str) -> float:
    """Parse the exponent spellings used by Fortran-based CalculiX builds."""
    normalized = token.strip().replace("D", "E").replace("d", "e")
    try:
        return float(normalized)
    except ValueError:
        # Some fixed-width Fortran writers omit the ``E`` (``1.2-003``).
        for index in range(len(normalized) - 1, 0, -1):
            if normalized[index] in "+-" and normalized[index - 1].isdigit():
                return float(normalized[:index] + "E" + normalized[index:])
        raise


def _parse_data_record(line: str) -> tuple[int, list[float]]:
    """Parse one ``-1 <node_id> <v1> <v2> ...`` data line.

    CCX writes FRD in **fixed-column** format (KEY(3) + NID(I10) + values
    in E12.5).  Adjacent values that are both negative look like
    ``-5.00000E+01-5.00000E+01`` with no separator, which breaks naive
    whitespace splitting — the split sees one giant blob and the node ends
    up with too few values (and the row gets silently dropped downstream).

    So always prefer fixed-column slicing.  We only fall back to whitespace
    splitting on truncated lines that are too short for the column layout.
    """
    body = line.rstrip("\n").rstrip("\r")
    if len(body) >= 13 and body[:3].strip() == "-1":
        try:
            node_id = int(body[3:13])
        except ValueError:
            node_id = -1
        if node_id > 0:
            values: list[float] = []
            offset = 13
            try:
                while offset + 12 <= len(body):
                    chunk = body[offset : offset + 12].strip()
                    if chunk:
                        values.append(_to_float(chunk))
                    offset += 12
                # Trailing partial column (some builds use a partial final field).
                if offset < len(body):
                    tail_chunk = body[offset:].strip()
                    if tail_chunk:
                        values.append(_to_float(tail_chunk))
            except ValueError:
                values = []
            if values:
                return node_id, values

    # Whitespace fallback for short / non-fixed-format files.
    parts = body.split()
    if len(parts) >= 2 and parts[0] == "-1":
        try:
            node_id = int(parts[1])
        except ValueError:
            return -1, []
        values = []
        for token in parts[2:]:
            try:
                values.append(float(token))
            except ValueError:
                pass
        return node_id, values

    return -1, []


def _parse_node_block(
    lines: list[str], start: int
) -> tuple[dict[int, tuple[float, float, float]], int]:
    """Read a nodal-coordinate block (header ``2C``) returning {nid: (x,y,z)}."""
    coords: dict[int, tuple[float, float, float]] = {}
    idx = start
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if stripped.startswith("-3"):
            return coords, idx + 1
        if stripped.startswith("-1"):
            nid, values = _parse_data_record(line)
            if nid > 0 and len(values) >= 3:
                coords[nid] = (values[0], values[1], values[2])
        idx += 1
    return coords, idx


def _parse_result_block(
    lines: list[str], start: int, num_components: int
) -> tuple[dict[int, list[float]], int]:
    """Read per-node data rows until the closing ``-3`` record.

    CCX writes the ``-4`` header with N+1 components when it includes the
    synthetic ``ALL`` channel.  Data lines only carry the physical components,
    so accept any record that yields at least one value and store up to
    ``num_components`` of them.
    """
    data: dict[int, list[float]] = {}
    idx = start
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if stripped.startswith("-3"):
            return data, idx + 1
        if stripped.startswith("-1"):
            nid, values = _parse_data_record(line)
            if nid > 0 and values:
                data[nid] = values[:num_components] if num_components > 0 else values
        idx += 1
    return data, idx


def read_frd(path: str | Path) -> FrdResult:
    """Parse a CalculiX FRD file and return the displacement and stress fields.

    Returns
    -------
    dict
        ``nodes``       : (N, 3) array of nodal coordinates in CCX node order.
        ``node_ids``    : (N,) array of 1-based CCX node ids in the same order.
        ``displacement``: (N, 3) array of nodal displacements (last step found).
        ``stress``      : (N, 6) array of nodal Cauchy stress in
                          [SXX, SYY, SZZ, SXY, SYZ, SZX] order, when present.
        ``von_mises``   : (N,) nodal Von Mises stress derived from ``stress``.
        ``nodal_force`` : (N, 3) nodal/reaction force vectors, when present.
        ``ener``        : (N,) nodal strain-energy density, when present.
        ``steps``       : metadata for every result block encountered.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    coords: dict[int, tuple[float, float, float]] = {}
    last_disp: dict[int, list[float]] = {}
    last_stress: dict[int, list[float]] = {}
    last_force: dict[int, list[float]] = {}
    last_ener: dict[int, list[float]] = {}
    steps: list[FrdStep] = []

    idx = 0
    current_block: str | None = None
    current_components: int = 0
    current_time: float = 0.0
    current_step: int = 0

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue

        if stripped.startswith("2C"):
            coords, idx = _parse_node_block(lines, idx + 1)
            continue

        if stripped.startswith("100C"):
            # 100C "LANALYSIS" VALUE NUMNOD ... — value carries the step time.
            tokens = stripped.split()
            try:
                current_time = float(tokens[2])
            except (IndexError, ValueError):
                current_time = 0.0
            current_step = len(steps) + 1
            idx += 1
            continue

        if stripped.startswith("-4"):
            tokens = stripped.split()
            current_block = tokens[1].upper() if len(tokens) >= 2 else ""
            try:
                current_components = int(tokens[2]) if len(tokens) >= 3 else 0
            except ValueError:
                current_components = 0
            idx += 1
            continue

        if stripped.startswith("-5"):
            # Component header line; nothing to read besides counting.
            idx += 1
            continue

        if stripped.startswith("-1") and current_block:
            num_components = current_components or _guess_component_count(current_block)
            block_data, idx = _parse_result_block(lines, idx, num_components)
            steps.append(
                {
                    "step": current_step,
                    "name": str(current_block),
                    "time": float(current_time),
                }
            )
            if current_block == "DISP":
                last_disp = block_data
            elif current_block == "STRESS":
                last_stress = block_data
            elif current_block == "FORC":
                last_force = block_data
            elif current_block == "ENER":
                last_ener = block_data
            current_block = None
            current_components = 0
            continue

        idx += 1

    if not coords:
        raise ValueError(f"FRD file {path} contained no nodal coordinates.")

    node_ids_sorted = sorted(coords.keys())
    node_ids = np.asarray(node_ids_sorted, dtype=int)
    n_nodes = node_ids.size
    nodes = np.zeros((n_nodes, 3), dtype=float)
    for i, nid in enumerate(node_ids_sorted):
        nodes[i] = coords[nid]

    displacement = np.zeros((n_nodes, 3), dtype=float)
    if last_disp:
        for i, nid in enumerate(node_ids_sorted):
            row = last_disp.get(nid)
            if row and len(row) >= 3:
                displacement[i, 0] = row[0]
                displacement[i, 1] = row[1]
                displacement[i, 2] = row[2]

    stress = np.zeros((n_nodes, 6), dtype=float)
    von_mises = np.zeros(n_nodes, dtype=float)
    if last_stress:
        for i, nid in enumerate(node_ids_sorted):
            row = last_stress.get(nid)
            if row and len(row) >= 6:
                stress[i] = row[:6]
        sxx, syy, szz, sxy, syz, szx = (stress[:, k] for k in range(6))
        von_mises = np.sqrt(
            0.5
            * (
                (sxx - syy) ** 2
                + (syy - szz) ** 2
                + (szz - sxx) ** 2
                + 6.0 * (sxy**2 + syz**2 + szx**2)
            )
        )

    ener = np.zeros(n_nodes, dtype=float)
    if last_ener:
        for i, nid in enumerate(node_ids_sorted):
            row = last_ener.get(nid)
            if row:
                ener[i] = float(row[0])

    nodal_force = np.zeros((n_nodes, 3), dtype=float)
    if last_force:
        for i, nid in enumerate(node_ids_sorted):
            row = last_force.get(nid)
            if row and len(row) >= 3:
                nodal_force[i] = row[:3]

    return {
        "nodes": nodes,
        "node_ids": node_ids,
        "displacement": displacement,
        "stress": stress,
        "von_mises": von_mises,
        "nodal_force": nodal_force,
        "ener": ener,
        "steps": steps,
    }


def _guess_component_count(block_name: str) -> int:
    """Fallback component count when CCX omits the value from the ``-4`` header."""
    return {
        "DISP": 3,
        "STRESS": 6,
        "TOSTRAIN": 6,
        "FORC": 3,
        "PE": 6,
        "ENER": 1,
        "TEMP": 1,
    }.get(block_name.upper(), 1)
