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
``cad.topopt``    ``com.cad.sim.topopt_voxel``   SIMP topology optimisation
================  =============================  =======================

Inputs are matched against ``NumberNode`` / ``VariableNode`` instances in the
``.cad`` graph whose ``exposed_name`` property equals the kwarg name, and
against named ``CadQueryCodeNode`` parameters.  The optional ``_settings``
mapping can also drive validated numeric material, mesh, load, impact, crash,
and topology-optimization properties discovered by the function-block UI. Results
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

import ast
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Public API surface — kept short on purpose.
__all__ = [
    "fea", "crash", "topopt", "CadResult", "clear_cache",
    "discover_override_controls",
]


# ──────────────────────────────────────────────────────────────────────
# Terminal solver identifiers
# ──────────────────────────────────────────────────────────────────────
_FEA_ID    = "com.cad.sim.solver"
_CRASH_ID  = "com.cad.sim.crash_solver"
_TOPOPT_IDS = ("com.cad.sim.topopt_voxel",)


# Numeric Design Studio controls that are safe and meaningful to drive from a
# system-model function block.  File paths, display options, face selections,
# and backend switches intentionally remain owned by the saved .cad study.
_OVERRIDEABLE_PROPERTIES = {
    "com.cad.sim.material": (
        "youngs_modulus", "poissons_ratio", "density", "yield_strength",
        "tangent_modulus",
    ),
    "com.cad.sim.crash_material": (
        "youngs_modulus", "poissons_ratio", "density", "yield_strength",
        "tangent_modulus", "failure_strain", "enable_fracture",
        "strain_rate_sensitive",
    ),
    "com.cad.sim.mesh": (
        "element_size", "refinement_size", "shell_thickness", "shell_nip",
    ),
    "com.cad.sim.constraint": (
        "displacement_x", "displacement_y", "displacement_z",
        "displacement_x_enabled", "displacement_y_enabled",
        "displacement_z_enabled",
    ),
    "com.cad.sim.load": (
        "force_x", "force_y", "force_z", "moment_x", "moment_y", "moment_z",
        "gravity_accel",
    ),
    "com.cad.sim.pressure_load": ("pressure",),
    "com.cad.sim.impact": (
        "velocity_x", "velocity_y", "velocity_z", "node_tolerance",
        "wall_friction", "wall_gap_mm",
    ),
    "com.cad.sim.crash_solver": (
        "end_time", "n_frames", "time_steps", "enable_mass_scaling",
        "damping_alpha", "damping_beta", "enable_corotation", "enable_contact",
        "contact_stiffness", "contact_thickness", "contact_update_interval",
        "mass_scaling_threshold", "impactor_mass_kg",
    ),
    "com.cad.sim.topopt_voxel": (
        "nelx", "nely", "nelz", "volfrac", "rmin", "penal",
        "density_cutoff", "max_iter", "tol", "convergence_patience",
        "stress_constraint", "yield_stress", "max_member_size_voxels",
        "pattern_repeat", "force_ix_frac", "force_iy_frac", "force_iz_frac",
        "force_dir_x", "force_dir_y", "force_dir_z", "force_magnitude",
    ),
}

_OVERRIDE_GROUPS = {
    "com.cad.sim.material": "Material",
    "com.cad.sim.crash_material": "Material",
    "com.cad.sim.mesh": "Mesh",
    "com.cad.sim.constraint": "Boundary condition",
    "com.cad.sim.load": "Load",
    "com.cad.sim.pressure_load": "Load",
    "com.cad.sim.impact": "Impact",
    "com.cad.sim.crash_solver": "Crash solver",
    "com.cad.sim.topopt_voxel": "Topology optimization",
}

_PROPERTY_LABELS = {
    "youngs_modulus": "Young's modulus",
    "poissons_ratio": "Poisson's ratio",
    "density": "Density",
    "yield_strength": "Yield strength",
    "tangent_modulus": "Tangent modulus",
    "failure_strain": "Failure strain",
    "element_size": "Element size",
    "refinement_size": "Refinement size",
    "shell_thickness": "Shell thickness",
    "shell_nip": "Shell integration points",
    "end_time": "End time",
    "n_frames": "Result frames",
    "time_steps": "Time steps",
    "volfrac": "Target material fraction",
    "rmin": "Filter radius",
    "penal": "SIMP penalty",
    "max_iter": "Maximum iterations",
    "tol": "Convergence tolerance",
    "force_magnitude": "Force magnitude",
    "impactor_mass_kg": "Impactor mass",
}


def _override_identifier(type_name: str) -> str | None:
    text = str(type_name or "")
    for identifier in sorted(_OVERRIDEABLE_PROPERTIES, key=len, reverse=True):
        if text == identifier or text.startswith(identifier + "."):
            return identifier
    return None


def discover_override_controls(session_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return material/mesh/load/solver controls available in a saved study.

    The returned ``key`` is stable for that saved graph and can be passed in
    the private ``_settings`` mapping of :func:`fea`, :func:`crash`, or
    :func:`topopt`.
    """
    controls: list[dict[str, Any]] = []
    for node_id, node_data in (session_data.get("nodes", {}) or {}).items():
        identifier = _override_identifier(node_data.get("type_", ""))
        if not identifier:
            continue
        custom = node_data.get("custom", {}) or {}
        node_name = str(node_data.get("name") or identifier.rsplit(".", 1)[-1])
        for prop in _OVERRIDEABLE_PROPERTIES[identifier]:
            value = custom.get(prop)
            if not isinstance(value, (bool, int, float)):
                continue
            controls.append({
                "key": f"{node_name}::{prop}",
                "node_id": str(node_id),
                "node": node_name,
                "group": _OVERRIDE_GROUPS.get(identifier, "Analysis setting"),
                "property": prop,
                "label": _PROPERTY_LABELS.get(prop, prop.replace("_", " ").title()),
                "value": value,
            })
    return controls


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
        peak_stress = raw.get("peak_stress_nodal", raw.get("max_stress_gauss", 0.0))
        std["max_stress"]    = float(peak_stress or 0.0)
        std["compliance"]    = float(raw.get("compliance") or 0.0)
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
        if "final_vol_frac" in raw:
            final_vol_frac = float(raw.get("final_vol_frac") or 0.0)
        else:
            elem_vol = raw.get("element_volumes", None)
            if density is not None and elem_vol is not None and len(density) == len(elem_vol):
                density_arr = np.asarray(density, dtype=float)
                elem_vol_arr = np.asarray(elem_vol, dtype=float)
                denom = float(np.sum(elem_vol_arr))
                final_vol_frac = (
                    float(np.sum(density_arr * elem_vol_arr) / denom)
                    if denom > 0.0 else 0.0
                )
            else:
                final_vol_frac = float(np.mean(density)) if density is not None and len(density) else 0.0
        std["final_vol_frac"] = final_vol_frac
        std["target_vol_frac"] = float(raw.get("target_vol_frac", 0.0))
        std["compliance"]     = float(raw.get("compliance", 0.0))
        std["mass"]           = float(raw.get("mass", 0.0))
        std["volume"]         = float(raw.get("volume", 0.0))
        std["total_volume"]   = float(raw.get("total_volume", 0.0))

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
def fea(cad_path: str, _settings: Mapping[str, Any] | None = None, **inputs) -> CadResult:
    """Run the FEA-solver path of a CAD graph and return its scalar results."""
    return _evaluate(cad_path, inputs, terminal_id=_FEA_ID, kind="fea", settings=_settings)


def crash(cad_path: str, _settings: Mapping[str, Any] | None = None, **inputs) -> CadResult:
    """Run the crash-solver path of a CAD graph and return its scalar results."""
    return _evaluate(cad_path, inputs, terminal_id=_CRASH_ID, kind="crash", settings=_settings)


def topopt(cad_path: str, _settings: Mapping[str, Any] | None = None, **inputs) -> CadResult:
    """Run a SIMP topology-optimisation pass through a CAD graph."""
    return _evaluate(cad_path, inputs, terminal_id=_TOPOPT_IDS, kind="topopt", settings=_settings)


# ──────────────────────────────────────────────────────────────────────
# Core driver
# ──────────────────────────────────────────────────────────────────────
def _evaluate(
    cad_path: str,
    inputs: Mapping[str, Any],
    terminal_id: str | Sequence[str],
    kind: str,
    settings: Mapping[str, Any] | None = None,
) -> CadResult:
    abs_path = os.path.abspath(str(cad_path))
    if not os.path.isfile(abs_path):
        # Fallback: resolve repo-relative paths against the PyLCSS repo root
        # so that saved system models can reference `data/foo.cad` portably.
        try:
            from pylcss.config import BASE_DIR
            repo_relative = os.path.join(os.path.dirname(BASE_DIR), str(cad_path))
            if os.path.isfile(repo_relative):
                abs_path = os.path.abspath(repo_relative)
        except Exception:
            pass
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"CAD graph file not found: {abs_path}")

    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        mtime = 0.0

    canonical_inputs = tuple(sorted((str(k), _to_float(v)) for k, v in inputs.items()))
    canonical_settings = tuple(
        sorted((str(k), _to_float(v)) for k, v in (settings or {}).items())
    )
    cache_key = (abs_path, mtime, kind, canonical_inputs, canonical_settings)

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
    _apply_property_overrides(graph, dict(canonical_settings))

    from pylcss.design_studio.engine import execute_graph
    execute_graph(graph)

    terminal_result = _find_terminal_result(graph, terminal_id)
    if terminal_result is None:
        expected = (
            "', '".join(terminal_id)
            if isinstance(terminal_id, (list, tuple))
            else str(terminal_id)
        )
        raise RuntimeError(
            f"CAD graph {abs_path!r} produced no result for terminal node "
            f"'{expected}'. Add the expected solver/optimisation node to the graph."
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
    from pylcss.design_studio.node_library import NODE_CLASS_MAPPING

    graph = NodeGraph()
    for node_class in NODE_CLASS_MAPPING.values():
        try:
            graph.register_node(node_class)
        except Exception as exc:
            logger.warning(
                "Could not register CAD node %s while loading %s: %s",
                getattr(node_class, "__name__", node_class), abs_path, exc,
            )

    with open(abs_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    graph.clear_session()
    graph.deserialize_session(session_data)
    return graph


def _apply_exposed_inputs(graph, inputs: Mapping[str, float]) -> tuple[int, set]:
    """Push kwargs into named CAD graph parameters.

    Supported targets:
    - NumberNode / VariableNode via ``exposed_name`` (or VariableNode name).
    - CadQueryCodeNode parameter slots ``param_1_name`` ... ``param_6_name``.
    - CadQueryCodeNode extra parameter text entries written as ``name=value``.

    Returns ``(applied_count, available_names_set)``.  The caller can compare
    requested vs. available to surface a clear KeyError for typos.
    """
    available: set = set()
    applied = 0
    for node in graph.all_nodes():
        if not hasattr(node, "has_property"):
            continue

        if node.has_property("exposed_name"):
            ename = (node.get_property("exposed_name") or "").strip()
            # Fall back to ``variable_name`` for VariableNode when exposed_name is blank.
            if not ename and node.has_property("variable_name"):
                ename = (node.get_property("variable_name") or "").strip()
            if ename:
                available.add(ename)
                if ename in inputs:
                    value = float(inputs[ename])
                    try:
                        if node.has_property("value_input"):
                            node.set_property("value_input", repr(value))
                        if node.has_property("value"):
                            node.set_property("value", value)
                    except Exception as exc:
                        logger.warning(
                            "cad runtime: failed to set %s=%s: %s", ename, value, exc
                        )
                    else:
                        _mark_node_dirty(node)
                        applied += 1

        if _is_code_part_node(node):
            applied += _apply_code_part_inputs(node, inputs, available)

    return applied, available


def _apply_property_overrides(graph, settings: Mapping[str, float]) -> int:
    """Apply validated ``node_name::property`` overrides to a fresh CAD graph."""
    if not settings:
        return 0

    nodes_by_name = {}
    for node in graph.all_nodes():
        node_name = node.name() if hasattr(node, "name") else ""
        nodes_by_name[str(node_name)] = node

    applied = 0
    for key, numeric_value in settings.items():
        if "::" not in key:
            raise KeyError(
                f"Invalid Design Studio setting key {key!r}; expected 'node_name::property'."
            )
        node_name, prop = key.rsplit("::", 1)
        node = nodes_by_name.get(node_name)
        if node is None:
            raise KeyError(f"Design Studio node {node_name!r} no longer exists in the saved study.")

        identifier = _override_identifier(getattr(node, "__identifier__", ""))
        allowed = _OVERRIDEABLE_PROPERTIES.get(identifier or "", ())
        if prop not in allowed or not node.has_property(prop):
            node_name = node.name() if hasattr(node, "name") else node_name
            raise KeyError(
                f"Setting {prop!r} on {node_name!r} is not an externally controllable numeric setting."
            )

        current = node.get_property(prop)
        if isinstance(current, bool):
            value = bool(round(float(numeric_value)))
        elif isinstance(current, int):
            value = int(round(float(numeric_value)))
        else:
            value = float(numeric_value)
        node.set_property(prop, value)
        _mark_node_dirty(node)
        applied += 1
    return applied


def _is_code_part_node(node) -> bool:
    return (
        getattr(node, "__identifier__", "") == "com.cad.code_part"
        or node.__class__.__name__ == "CadQueryCodeNode"
    )


def _mark_node_dirty(node) -> None:
    # Force re-execution: bust the engine's per-node dirty-state cache.
    setattr(node, "_last_result", None)
    setattr(node, "_last_input_hash", None)
    setattr(node, "_dirty", True)
    setattr(node, "_force_execute", True)


_PARAM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_code_part_parameters(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, dict):
            return {}
        return {str(k): v for k, v in parsed.items()}

    params: Dict[str, Any] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if _PARAM_NAME_RE.match(name):
            params[name] = value.strip()
    return params


def _format_code_part_parameters(params: Mapping[str, Any]) -> str:
    lines = []
    for name, value in params.items():
        if isinstance(value, str):
            try:
                float(value)
                text = value
            except ValueError:
                text = repr(value)
        else:
            text = repr(float(value))
        lines.append(f"{name}={text}")
    return "\n".join(lines)


def _apply_code_part_inputs(node, inputs: Mapping[str, float], available: set) -> int:
    applied = 0

    for idx in range(1, 7):
        name_prop = f"param_{idx}_name"
        value_prop = f"param_{idx}_value"
        if not node.has_property(name_prop) or not node.has_property(value_prop):
            continue
        pname = str(node.get_property(name_prop) or "").strip()
        if not pname:
            continue
        available.add(pname)
        if pname not in inputs:
            continue
        try:
            node.set_property(value_prop, float(inputs[pname]))
        except Exception as exc:
            logger.warning("cad runtime: failed to set code parameter %s: %s", pname, exc)
        else:
            _mark_node_dirty(node)
            applied += 1

    if not node.has_property("parameters"):
        return applied

    try:
        params = _parse_code_part_parameters(node.get_property("parameters") or "")
    except Exception as exc:
        logger.warning("cad runtime: failed to parse extra code parameters: %s", exc)
        return applied

    if not params:
        return applied

    available.update(params.keys())
    changed = False
    for name in list(params.keys()):
        if name not in inputs:
            continue
        params[name] = float(inputs[name])
        changed = True
        applied += 1

    if changed:
        node.set_property("parameters", _format_code_part_parameters(params))
        _mark_node_dirty(node)

    return applied


def _find_terminal_result(graph, terminal_id: str | Sequence[str]):
    """Pick the *last-executed* node whose identifier matches.

    A graph may legitimately contain more than one solver node (e.g. two FEA
    configurations).  Convention: the one farther downstream wins — that's the
    one with the most upstream-connected inputs that produced a result.
    """
    terminal_ids = (
        {terminal_id}
        if isinstance(terminal_id, str)
        else {str(item) for item in terminal_id}
    )
    candidates = [
        n for n in graph.all_nodes()
        if getattr(n, "__identifier__", "") in terminal_ids
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
