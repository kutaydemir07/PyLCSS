# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Headless CAD-graph evaluator exposed to system-modeling function blocks.

The compiled code of a sysmod ``CustomBlockNode`` sees this module bound to the
name ``cad`` (see :mod:`pylcss.system_modeling.model_builder`).  A function
block can then write::

    r = cad.fea("front_panel.cad", thickness=t, fillet_r=ro)
    return r.max_stress, r.mass

Three entry points are provided, each one targeting a different terminal solver
node inside the named ``.cad`` graph:

================  =============================  =======================
function          terminal node identifier       backend
================  =============================  =======================
``cad.fea``       ``com.cad.sim.solver``         CalculiX (linear static)
``cad.crash``     ``com.cad.sim.crash_solver``   OpenRadioss
``cad.topopt``    ``com.cad.sim.topopt``         CalculiX (SIMP)
================  =============================  =======================

Inputs are matched against ``NumberNode`` / ``VariableNode`` instances in the
``.cad`` graph whose ``exposed_name`` property equals the kwarg name.  Results
are wrapped in :class:`CadResult`, which gives attribute *and* dict access plus
a small fixed-name standard subset (``max_stress``, ``compliance``, ``mass``,
``volume``, ``peak_disp``, …) so user code is stable across graph versions.

Evaluations are cached on ``(absolute_path, mtime, kind, sorted_inputs)`` for
the lifetime of the running Python process.  Identical inputs never re-solve;
this is also the layer a surrogate model plugs into (a function block's
``use_surrogate`` checkbox short-circuits the call entirely before it reaches
here).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Public API surface — kept short on purpose.
__all__ = ["fea", "crash", "topopt", "CadResult", "clear_cache"]


# ──────────────────────────────────────────────────────────────────────
# Terminal solver identifiers
# ──────────────────────────────────────────────────────────────────────
_FEA_ID    = "com.cad.sim.solver"
_CRASH_ID  = "com.cad.sim.crash_solver"
_TOPOPT_ID = "com.cad.sim.topopt"


# ──────────────────────────────────────────────────────────────────────
# Result wrapper
# ──────────────────────────────────────────────────────────────────────
class CadResult:
    """Standardised view of a CAD-graph evaluation result.

    Standard fields are always present (filled with ``0.0`` / ``None`` when the
    underlying solver did not emit them).  The raw result dict from the
    terminal node remains accessible via attribute / item lookup, so anything
    the solver produces — VTK-renderable mesh, ENER fields, FRD step list — is
    still reachable when a function-block needs it.
    """

    __slots__ = ("_kind", "_raw", "_standard")

    def __init__(self, kind: str, raw: Mapping[str, Any]):
        self._kind = str(kind)
        self._raw = dict(raw) if raw is not None else {}
        self._standard = _standardize(self._kind, self._raw)

    # -- access -------------------------------------------------------
    def __getattr__(self, name: str):
        std = object.__getattribute__(self, "_standard")
        raw = object.__getattribute__(self, "_raw")
        if name in std:
            return std[name]
        if name in raw:
            return raw[name]
        raise AttributeError(
            f"CadResult has no field '{name}'. "
            f"Standard fields: {sorted(std)}; raw keys: {sorted(raw)}"
        )

    def __getitem__(self, key: str):
        return self.__getattr__(key)

    def __contains__(self, key: str) -> bool:
        return key in self._standard or key in self._raw

    def pick(self, *names: str) -> tuple:
        """Return the requested fields in order — for tuple unpacking.

        Example::

            s, m = cad.fea("p.cad", t=2.5).pick("max_stress", "mass")
        """
        return tuple(self[n] for n in names)

    def standard(self) -> Dict[str, Any]:
        """Return a fresh dict of the standardised fields."""
        return dict(self._standard)

    def raw(self) -> Dict[str, Any]:
        """Return the raw underlying result dict (mesh, fields, FRD path, …)."""
        return dict(self._raw)

    @property
    def kind(self) -> str:
        return self._kind

    def __repr__(self) -> str:
        scalars = {k: v for k, v in self._standard.items()
                   if isinstance(v, (int, float, str, type(None)))}
        return f"CadResult(kind={self._kind!r}, {scalars})"


def _standardize(kind: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Map the solver-specific result keys onto the documented standard set."""
    std: Dict[str, Any] = {}

    if kind == "fea":
        std["max_stress"]    = float(raw.get("max_stress_gauss", 0.0))
        std["compliance"]    = float(raw.get("compliance", 0.0))
        std["strain_energy"] = float(raw.get("strain_energy", 0.0))
        std["volume"]        = float(raw.get("volume", 0.0))
        std["mass"]          = float(raw.get("mass", 0.0))
        std["peak_disp"]     = float(raw.get("peak_displacement", 0.0))

    elif kind == "crash":
        std["max_stress"]      = float(raw.get("peak_stress", 0.0))
        std["peak_disp"]       = float(raw.get("peak_displacement", 0.0))
        std["absorbed_energy"] = float(raw.get("absorbed_energy", 0.0))
        std["n_failed"]        = int(raw.get("n_failed", 0))

    elif kind == "topopt":
        density = raw.get("density", None)
        std["final_vol_frac"] = float(np.mean(density)) if density is not None and len(density) else 0.0
        std["compliance"]     = float(raw.get("compliance", 0.0))
        std["mass"]           = float(raw.get("mass", 0.0))
        std["volume"]         = float(raw.get("volume", 0.0))

    return std


# ──────────────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────────────
_cache: Dict[tuple, CadResult] = {}
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Drop every cached CAD-graph evaluation (per-process)."""
    with _cache_lock:
        _cache.clear()


# ──────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────
def fea(cad_path: str, **inputs) -> CadResult:
    """Run the FEA-solver path of a CAD graph and return its scalar results."""
    return _evaluate(cad_path, inputs, terminal_id=_FEA_ID, kind="fea")


def crash(cad_path: str, **inputs) -> CadResult:
    """Run the crash-solver path of a CAD graph and return its scalar results."""
    return _evaluate(cad_path, inputs, terminal_id=_CRASH_ID, kind="crash")


def topopt(cad_path: str, **inputs) -> CadResult:
    """Run a SIMP topology-optimisation pass through a CAD graph."""
    return _evaluate(cad_path, inputs, terminal_id=_TOPOPT_ID, kind="topopt")


# ──────────────────────────────────────────────────────────────────────
# Core driver
# ──────────────────────────────────────────────────────────────────────
def _evaluate(cad_path: str, inputs: Mapping[str, Any], terminal_id: str, kind: str) -> CadResult:
    abs_path = os.path.abspath(str(cad_path))
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"CAD graph file not found: {abs_path}")

    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        mtime = 0.0

    canonical_inputs = tuple(sorted((str(k), _to_float(v)) for k, v in inputs.items()))
    cache_key = (abs_path, mtime, kind, canonical_inputs)

    with _cache_lock:
        cached = _cache.get(cache_key)
    if cached is not None:
        logger.debug("cad runtime: cache hit %s %s", kind, abs_path)
        return cached

    logger.info(
        "cad runtime: evaluating %s on %s with %d input(s)",
        kind, abs_path, len(canonical_inputs),
    )

    _ensure_qapp()
    graph = _load_graph(abs_path)

    set_count, available_names = _apply_exposed_inputs(graph, dict(canonical_inputs))
    missing = set(dict(canonical_inputs).keys()) - {k for k, _ in canonical_inputs[:set_count]}
    # The above ‘missing’ is computed inside _apply_exposed_inputs already; recompute defensively.
    requested = {k for k, _ in canonical_inputs}
    if not requested.issubset(available_names):
        missing = sorted(requested - available_names)
        raise KeyError(
            f"CAD graph {abs_path!r} has no exposed parameters named {missing}. "
            f"Available: {sorted(available_names)}"
        )

    from pylcss.cad.engine import execute_graph
    execute_graph(graph)

    terminal_result = _find_terminal_result(graph, terminal_id)
    if terminal_result is None:
        raise RuntimeError(
            f"CAD graph {abs_path!r} produced no result for terminal node "
            f"'{terminal_id}'. Add the expected solver/optimisation node to the graph."
        )

    wrapped = CadResult(kind, terminal_result)
    with _cache_lock:
        _cache[cache_key] = wrapped
    return wrapped


# ──────────────────────────────────────────────────────────────────────
# Graph helpers
# ──────────────────────────────────────────────────────────────────────
def _load_graph(abs_path: str):
    """Spin up a fresh ``NodeGraph``, register every CAD node, deserialise the file."""
    from NodeGraphQt import NodeGraph
    from pylcss.cad.node_library import NODE_CLASS_MAPPING

    graph = NodeGraph()
    for node_class in NODE_CLASS_MAPPING.values():
        try:
            graph.register_node(node_class)
        except Exception:
            # NodeGraphQt rejects re-registration silently in most builds; ignore.
            pass

    with open(abs_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    graph.clear_session()
    graph.deserialize_session(session_data)
    return graph


def _apply_exposed_inputs(graph, inputs: Mapping[str, float]) -> tuple[int, set]:
    """Push the kwargs into matching ``exposed_name`` nodes.

    Returns ``(applied_count, available_names_set)``.  The caller can compare
    requested vs. available to surface a clear KeyError for typos.
    """
    available: set = set()
    applied = 0
    for node in graph.all_nodes():
        if not hasattr(node, "has_property"):
            continue
        if not node.has_property("exposed_name"):
            continue
        ename = (node.get_property("exposed_name") or "").strip()
        # Fall back to ``variable_name`` for VariableNode when exposed_name is blank.
        if not ename and node.has_property("variable_name"):
            ename = (node.get_property("variable_name") or "").strip()
        if not ename:
            continue
        available.add(ename)
        if ename not in inputs:
            continue
        value = float(inputs[ename])
        try:
            if node.has_property("value_input"):
                node.set_property("value_input", repr(value))
            if node.has_property("value"):
                node.set_property("value", value)
        except Exception as exc:
            logger.warning("cad runtime: failed to set %s=%s: %s", ename, value, exc)
            continue
        # Force re-execution: bust the engine's per-node dirty-state cache.
        setattr(node, "_last_result", None)
        setattr(node, "_last_input_hash", None)
        setattr(node, "_dirty", True)
        setattr(node, "_force_execute", True)
        applied += 1
    return applied, available


def _find_terminal_result(graph, terminal_id: str):
    """Pick the *last-executed* node whose identifier matches.

    A graph may legitimately contain more than one solver node (e.g. two FEA
    configurations).  Convention: the one farther downstream wins — that's the
    one with the most upstream-connected inputs that produced a result.
    """
    candidates = [
        n for n in graph.all_nodes()
        if getattr(n, "__identifier__", "") == terminal_id
    ]
    if not candidates:
        return None

    def _depth(node) -> int:
        count = 0
        if not hasattr(node, "input_ports"):
            return 0
        ports = node.input_ports()
        if isinstance(ports, dict):
            ports = list(ports.values())
        for port in ports:
            if not hasattr(port, "connected_ports"):
                continue
            for cp in port.connected_ports():
                count += 1
                up = cp.node()
                count += _depth(up) if up is not node else 0
        return count

    candidates.sort(key=_depth, reverse=True)
    for node in candidates:
        result = getattr(node, "_last_result", None)
        if result is not None:
            return result
    return None


# ──────────────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────────────
def _ensure_qapp():
    """NodeGraphQt requires a QApplication. The sysmod GUI already has one; this
    is a defensive fallback for headless test/script contexts."""
    try:
        from qtpy import QtWidgets
    except Exception:
        try:
            from PySide6 import QtWidgets  # type: ignore
        except Exception:
            from PyQt5 import QtWidgets    # type: ignore
    app = QtWidgets.QApplication.instance()
    if app is None:
        import sys
        app = QtWidgets.QApplication(sys.argv if hasattr(sys, "argv") else [])
    return app


def _to_float(value: Any) -> float:
    """Coerce kwarg values so cache keys are hashable & stable."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
