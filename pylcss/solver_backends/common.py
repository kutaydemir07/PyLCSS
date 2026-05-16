# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Shared helpers for external solver integrations."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Cache the solver_paths.json so we hit the disk at most once per process.
# `None` means "not loaded yet"; `{}` means "loaded, nothing found".
_SOLVER_PATHS_CACHE: Optional[Dict[str, str]] = None


def _solver_paths_config_path() -> Path:
    """Location of the JSON config written by scripts/install_solvers.py."""
    # pylcss/solver_backends/common.py -> repo root is two parents up.
    return Path(__file__).resolve().parent.parent.parent / "external_solvers" / "solver_paths.json"


def _load_solver_paths_config() -> Dict[str, str]:
    """Lazy-load the JSON config of pre-resolved solver executable paths."""
    global _SOLVER_PATHS_CACHE
    if _SOLVER_PATHS_CACHE is not None:
        return _SOLVER_PATHS_CACHE
    config_path = _solver_paths_config_path()
    if not config_path.is_file():
        _SOLVER_PATHS_CACHE = {}
        return _SOLVER_PATHS_CACHE
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        _SOLVER_PATHS_CACHE = {}
        return _SOLVER_PATHS_CACHE
    if not isinstance(raw, dict):
        _SOLVER_PATHS_CACHE = {}
        return _SOLVER_PATHS_CACHE
    _SOLVER_PATHS_CACHE = {str(k): str(v) for k, v in raw.items() if v}
    return _SOLVER_PATHS_CACHE


class SolverBackendError(RuntimeError):
    """Raised when an external solver backend cannot prepare or run a case."""


@dataclass
class ExternalRunConfig:
    """Runtime options shared by external solver adapters."""

    executable: Optional[str] = None
    secondary_executable: Optional[str] = None
    work_dir: Optional[str] = None
    keep_files: bool = True
    run_solver: bool = False
    timeout_s: float = 3600.0
    job_name: str = "pylcss_case"


def flatten_inputs(items: Iterable[Any]) -> List[Any]:
    """Flatten nested node-graph input lists while preserving non-None values."""
    result: List[Any] = []
    for item in items or []:
        if isinstance(item, list):
            result.extend(flatten_inputs(item))
        elif item is not None:
            result.append(item)
    return result


def as_bool(value: Any) -> bool:
    """Interpret bool-like UI property values safely."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "checked"}
    return bool(value)


def make_work_dir(prefix: str, requested_dir: Optional[str]) -> Path:
    """Create or reuse a working directory for generated solver artifacts."""
    if requested_dir:
        path = Path(os.path.expandvars(os.path.expanduser(str(requested_dir))))
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix=prefix))


def resolve_executable(
    explicit: Optional[str],
    env_vars: Sequence[str],
    candidates: Sequence[str],
) -> Optional[str]:
    """Resolve an executable from an explicit path, the JSON solver config,
    environment vars, or PATH.

    Lookup order (first hit wins):
        1. ``explicit`` — value passed in from a node property.
        2. ``external_solvers/solver_paths.json`` written by
           ``scripts/install_solvers.py``.  Looked up by each name in
           ``env_vars`` so the same key namespace works for both stores.
        3. Real OS environment variables in ``env_vars``.
        4. ``candidates`` resolved via ``shutil.which``.
    """
    probes: List[str] = []
    if explicit:
        probes.append(str(explicit))

    config = _load_solver_paths_config()
    for env_name in env_vars:
        config_val = config.get(env_name)
        if config_val:
            probes.append(config_val)

    for env_name in env_vars:
        env_val = os.environ.get(env_name)
        if env_val:
            probes.append(env_val)

    probes.extend(candidates)

    for probe in probes:
        expanded = os.path.expandvars(os.path.expanduser(probe))
        if os.path.isfile(expanded):
            return expanded
        found = shutil.which(expanded)
        if found:
            return found
    return None


def run_process(
    args: Sequence[str],
    cwd: Path,
    timeout_s: float,
    extra_path_dirs: Sequence[str] = (),
    extra_env: Optional[dict] = None,
    stdout_file: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run an external solver process and capture text output.

    ``extra_path_dirs`` is prepended to ``PATH`` for the child only — needed on
    Windows where CalculiX / OpenRadioss ship sibling DLLs that the loader
    must be able to find for the .exe to start at all (otherwise Windows
    returns exit code 0xC0000135 with no stdout).

    ``extra_env`` adds / overrides individual environment variables (e.g.
    ``RAD_CFG_PATH`` for OpenRadioss).

    ``stdout_file`` redirects stdout+stderr to a file on disk and avoids the
    Python subprocess pipe-buffer deadlock that hits any solver which prints
    more than the Windows pipe buffer (~4 KB) of progress info during its run.
    The returned ``CompletedProcess.stdout`` is then populated from that file
    so callers see the same string they'd otherwise have gotten from PIPE.
    """
    env = None
    if extra_path_dirs or extra_env:
        env = os.environ.copy()
        if extra_path_dirs:
            sep = os.pathsep
            prepend = sep.join(str(d) for d in extra_path_dirs if d)
            if prepend:
                env["PATH"] = prepend + sep + env.get("PATH", "")
        if extra_env:
            for k, v in extra_env.items():
                env[str(k)] = str(v)

    if stdout_file is not None:
        stdout_file = Path(stdout_file)
        stdout_file.parent.mkdir(parents=True, exist_ok=True)
        # Start a watcher thread that periodically prints any new lines in the
        # log file to stdout — this gives the user visible solver progress
        # ("NC= 12300 T= 1.23E-04 ...") instead of staring at a frozen UI.
        import threading
        import time as _time

        stop_event = threading.Event()

        def _tail():
            """Emit ONE concise progress line every ``EMIT_INTERVAL`` seconds.

            Radioss spams a 2-line update every ~0.2 s (NC= ... + ELAPSED ...).
            Echoing each one floods the console.  Instead we read the file
            continuously, remember the latest NC and ELAPSED lines we've seen,
            and print only the pair once per interval.  We also emit a
            ``TERMINATION`` line immediately when one appears.
            """
            EMIT_INTERVAL = 30.0
            last_size = 0
            last_emit = 0.0
            latest_nc = ""
            latest_elapsed = ""
            while not stop_event.is_set():
                try:
                    cur = stdout_file.stat().st_size
                except OSError:
                    _time.sleep(0.5)
                    continue
                if cur > last_size:
                    try:
                        with open(stdout_file, "r", encoding="utf-8", errors="replace") as r:
                            r.seek(last_size)
                            tail_chunk = r.read()
                    except OSError:
                        tail_chunk = ""
                    last_size = cur
                    for line in tail_chunk.splitlines():
                        s = line.strip()
                        if s.startswith("NC="):
                            latest_nc = s
                        elif "ELAPSED TIME" in s:
                            latest_elapsed = s
                        elif "TERMINATION" in s:
                            # Show termination immediately, without throttling.
                            print(f"  | {s}")
                now = _time.time()
                if now - last_emit >= EMIT_INTERVAL and (latest_nc or latest_elapsed):
                    if latest_nc:
                        print(f"  | {latest_nc}")
                    if latest_elapsed:
                        print(f"  | {latest_elapsed}")
                    last_emit = now
                stop_event.wait(2.0)

        watcher = threading.Thread(target=_tail, daemon=True)
        watcher.start()
        try:
            with open(stdout_file, "w", encoding="utf-8", errors="replace") as fout:
                proc = subprocess.run(
                    list(args),
                    cwd=str(cwd),
                    text=True,
                    stdout=fout,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    check=False,
                    env=env,
                )
        finally:
            stop_event.set()
            watcher.join(timeout=2.0)
        try:
            proc.stdout = stdout_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            proc.stdout = ""
        return proc

    return subprocess.run(
        list(args),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
        env=env,
    )


def mesh_to_tet4(mesh: Any, warnings: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return points and right-hand-oriented 4-node tetrahedra from a scikit-fem mesh.

    CalculiX (Abaqus C3D4) and OpenRadioss (LS-DYNA solid elform 10) both require
    the four corner nodes ``[n1, n2, n3, n4]`` to form a *positive* signed
    volume — i.e. ``(n2-n1) x (n3-n1)`` must point toward ``n4``.  skfem /
    netgen do not guarantee that orientation, so we flip any tet whose signed
    volume is negative by swapping nodes 0 and 1.  Without this, ccx fails
    with "*ERROR in e_c3d: nonpositive jacobian determinant" on roughly half
    the elements.
    """
    if mesh is None or not hasattr(mesh, "p") or not hasattr(mesh, "t"):
        raise SolverBackendError("External solver backend expected a scikit-fem mesh.")

    points = np.asarray(mesh.p, dtype=float).T            # (N_nodes, 3)
    cells = np.asarray(mesh.t, dtype=int).T                # (N_elem, ≥4)
    if cells.ndim != 2 or cells.shape[1] < 4:
        raise SolverBackendError("Only tetrahedral solid meshes are currently supported.")
    if cells.shape[1] > 4:
        warnings.append(
            "Higher-order tetrahedra were downgraded to their first four corner nodes "
            "for the external deck writer."
        )
        cells = cells[:, :4].copy()
    else:
        cells = cells.copy()

    # Signed volume per tet: V = (1/6) * det([n1-n0, n2-n0, n3-n0]).
    v0 = points[cells[:, 0]]
    v1 = points[cells[:, 1]]
    v2 = points[cells[:, 2]]
    v3 = points[cells[:, 3]]
    signed_vol = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0)

    flipped = signed_vol < 0.0
    n_flipped = int(np.count_nonzero(flipped))
    if n_flipped:
        # Swapping any two corners inverts orientation; swap n0/n1.
        cells[flipped, 0], cells[flipped, 1] = cells[flipped, 1], cells[flipped, 0].copy()
        warnings.append(
            f"Reoriented {n_flipped}/{cells.shape[0]} tetrahedra with negative "
            "signed volume so CalculiX/OpenRadioss accept them as C3D4."
        )

    degenerate = np.isclose(signed_vol, 0.0)
    n_deg = int(np.count_nonzero(degenerate))
    if n_deg:
        warnings.append(
            f"{n_deg} tetrahedra have a near-zero signed volume — the input mesh "
            "is degenerate and the solver may still reject it."
        )

    return points, cells


def id_lines(ids: Sequence[int], per_line: int = 12) -> List[str]:
    """Format 1-based solver ids into comma-separated deck lines."""
    values = [int(v) for v in ids]
    lines: List[str] = []
    for i in range(0, len(values), per_line):
        lines.append(", ".join(str(v) for v in values[i : i + per_line]))
    return lines


def load_vector(load: dict) -> np.ndarray:
    """Return a 3-vector from a load dictionary."""
    return np.asarray(load.get("vector", [0.0, 0.0, 0.0]), dtype=float)


def normalize_geometries(value: Any) -> List[Any]:
    """Normalize node outputs and dictionaries into a list of CadQuery faces."""
    if value is None:
        return []
    if isinstance(value, dict):
        faces = value.get("geometries") or value.get("faces")
        if faces:
            return [f for f in faces if f is not None]
        face = value.get("geometry") or value.get("face")
        return [face] if face is not None else []
    if isinstance(value, (list, tuple)):
        return [f for f in value if f is not None]
    return [value]


def dict_geometries(data: dict) -> List[Any]:
    """Extract geometry references from constraint/load dictionaries."""
    geoms = data.get("geometries", None)
    if geoms is None:
        geoms = data.get("geometry", None)
    return normalize_geometries(geoms)


_ALLOWED_CONDITION_FUNCS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
    "isclose": np.isclose,
}
_ALLOWED_NP_ATTRS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
    "isclose": np.isclose,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "logical_not": np.logical_not,
    "pi": np.pi,
}


class _SafeNumpy:
    """Tiny namespace for condition expressions such as ``np.abs(z) < 1``."""

    def __getattr__(self, name: str) -> Any:
        if name not in _ALLOWED_NP_ATTRS:
            raise AttributeError(name)
        return _ALLOWED_NP_ATTRS[name]


class _ConditionValidator(ast.NodeVisitor):
    """Whitelist the expression subset used by legacy CAD condition strings."""

    _ALLOWED_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Call,
        ast.Attribute,
        ast.And,
        ast.Or,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Invert,
        ast.Not,
        ast.UAdd,
        ast.USub,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )
    _ALLOWED_NAMES = {"x", "y", "z", "np", *tuple(_ALLOWED_CONDITION_FUNCS)}

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self._ALLOWED_NODES):
            raise ValueError(f"unsupported expression element: {node.__class__.__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self._ALLOWED_NAMES:
            raise ValueError(f"unsupported name: {node.id!r}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.value, ast.Name) or node.value.id != "np":
            raise ValueError("only np.<function> attributes are allowed")
        if node.attr not in _ALLOWED_NP_ATTRS:
            raise ValueError(f"unsupported numpy function: np.{node.attr}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in _ALLOWED_CONDITION_FUNCS:
                raise ValueError(f"unsupported function: {node.func.id!r}")
        elif isinstance(node.func, ast.Attribute):
            self.visit_Attribute(node.func)
        else:
            raise ValueError("unsupported function call")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("condition expressions do not support **kwargs")
            self.visit(keyword.value)


def nodes_matching_condition(
    mesh: Any,
    condition: str,
    warnings: Optional[List[str]] = None,
    label: str = "condition",
) -> np.ndarray:
    """Find mesh node indices selected by a legacy ``x/y/z`` condition string."""
    expr = str(condition or "").strip()
    if not expr:
        return np.array([], dtype=int)
    if mesh is None or not hasattr(mesh, "p"):
        raise SolverBackendError("Condition-based selection expected a mesh with node coordinates.")

    p = np.asarray(mesh.p, dtype=float)
    x, y, z = p[0], p[1], p[2]
    try:
        expr_ast = ast.parse(expr, mode="eval")
        _ConditionValidator().visit(expr_ast)
        mask = eval(
            compile(expr_ast, "<pylcss-condition>", "eval"),
            {"__builtins__": {}},
            {
                "x": x,
                "y": y,
                "z": z,
                "np": _SafeNumpy(),
                **_ALLOWED_CONDITION_FUNCS,
            },
        )
    except Exception as exc:
        raise SolverBackendError(f"{label} expression {expr!r} could not be evaluated: {exc}") from exc

    mask = np.asarray(mask, dtype=bool)
    if mask.shape == ():
        mask = np.full(x.shape, bool(mask), dtype=bool)
    if mask.shape != x.shape:
        raise SolverBackendError(
            f"{label} expression {expr!r} returned shape {mask.shape}, expected {x.shape}."
        )

    node_ids = np.where(mask)[0].astype(int)
    if warnings is not None and node_ids.size == 0:
        warnings.append(f"{label} expression {expr!r} matched no mesh nodes.")
    return node_ids


def nodes_matching_geometries(
    mesh: Any,
    geometries: Sequence[Any],
    tolerance: float = 1.5,
) -> np.ndarray:
    """Find mesh node indices close to one or more CadQuery faces."""
    geoms = [g for g in geometries if g is not None]
    if not geoms:
        return np.array([], dtype=int)

    try:
        from cadquery import Vector
    except Exception:  # pragma: no cover - cadquery is expected in the app.
        Vector = None

    p = np.asarray(mesh.p, dtype=float)
    selected: List[int] = []
    for geom in geoms:
        candidates = range(p.shape[1])
        try:
            bb = geom.BoundingBox()
            mask = (
                (p[0] >= bb.xmin - tolerance)
                & (p[0] <= bb.xmax + tolerance)
                & (p[1] >= bb.ymin - tolerance)
                & (p[1] <= bb.ymax + tolerance)
                & (p[2] >= bb.zmin - tolerance)
                & (p[2] <= bb.zmax + tolerance)
            )
            candidates = np.where(mask)[0]
        except Exception:
            pass

        for node_idx in candidates:
            x, y, z = float(p[0, node_idx]), float(p[1, node_idx]), float(p[2, node_idx])
            matched = False
            if Vector is not None:
                try:
                    matched = geom.distanceTo(Vector(x, y, z)) <= tolerance
                except Exception:
                    matched = False
            if not matched:
                try:
                    bb = geom.BoundingBox()
                    matched = (
                        bb.xmin - tolerance <= x <= bb.xmax + tolerance
                        and bb.ymin - tolerance <= y <= bb.ymax + tolerance
                        and bb.zmin - tolerance <= z <= bb.zmax + tolerance
                    )
                except Exception:
                    matched = False
            if matched:
                selected.append(int(node_idx))

    return np.array(sorted(set(selected)), dtype=int)


def tail(text: str, limit: int = 4000) -> str:
    """Return the tail of a long solver log."""
    if len(text) <= limit:
        return text
    return text[-limit:]


def tet_face_sets_for_geometries(
    mesh: Any,
    geometries: Sequence[Any],
    tolerance: float = 1.5,
) -> List[Tuple[int, int]]:
    """Return ``(element_id_1based, face_local_id_1based)`` for every external
    tet face lying on one of the supplied CadQuery face geometries.

    CalculiX numbers tet faces 1..4 by the *opposite* corner node (face i is
    the triangle formed by the three corners excluding node i).  This is the
    convention used by ``*SURFACE, TYPE=ELEMENT`` and ``*DLOAD`` Pn loads.

    The function works on the boundary facets of the linear-tet topology in
    ``mesh.t[:4, :]`` so it is safe to call from both static and explicit code
    paths.
    """
    if mesh is None or not hasattr(mesh, "p") or not hasattr(mesh, "t"):
        return []

    p = np.asarray(mesh.p, dtype=float)             # (3, N_nodes)
    # Match the connectivity emitted by ``mesh_to_tet4``.  Pressure surfaces
    # use CalculiX local face ids, so their numbering must follow the same
    # reorientation used in the exported *ELEMENT block.
    try:
        _, cells = mesh_to_tet4(mesh, [])
        t = np.asarray(cells, dtype=int).T
    except Exception:
        t = np.asarray(mesh.t[:4, :], dtype=int)
    n_elem = t.shape[1]

    # Local face nodes — CCX C3D4 face numbering (from the *SURFACE / *DLOAD docs):
    #   Face 1: nodes 1-2-3  → 0-indexed positions {0, 1, 2}
    #   Face 2: nodes 1-4-2  → 0-indexed positions {0, 1, 3}
    #   Face 3: nodes 2-4-3  → 0-indexed positions {1, 2, 3}
    #   Face 4: nodes 3-4-1  → 0-indexed positions {0, 2, 3}
    # Row i of face_local gives the three 0-indexed positions for CCX face i+1,
    # so f_local+1 is already the correct CCX Sx face label to write in *SURFACE.
    face_local = np.array(
        [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=int
    )

    # Build a map: sorted 3-node tuple → list of (elem, face_local_1based).
    face_owners: dict = {}
    for elem in range(n_elem):
        elem_nodes = t[:, elem]
        for f_local in range(4):
            nodes = tuple(sorted(int(elem_nodes[k]) for k in face_local[f_local]))
            face_owners.setdefault(nodes, []).append((elem + 1, f_local + 1))

    # External faces: triangles owned by exactly one tet.
    external_faces = [(nodes, owners[0]) for nodes, owners in face_owners.items() if len(owners) == 1]
    if not external_faces:
        return []

    matching_nodes = nodes_matching_geometries(mesh, geometries, tolerance=tolerance)
    if matching_nodes.size == 0:
        return []
    matched_set = set(int(n) for n in matching_nodes.tolist())

    result: List[Tuple[int, int]] = []
    for nodes, (elem_id, face_id) in external_faces:
        if all(n in matched_set for n in nodes):
            result.append((elem_id, face_id))
    return result
