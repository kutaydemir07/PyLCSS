# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Headless CAD-graph evaluator exposed to system-modeling function blocks.

The compiled code of a sysmod ``CustomBlockNode`` sees this module bound to the
name ``cad`` (see :mod:`pylcss.mdao.system_modeling.compiler`).  A function
block can then write::

    r = cad.fea("front_panel.cad", thickness=t, fillet_r=ro)
    return r.max_stress, r.mass

Three primary entry points are provided, each one targeting a different terminal solver
node inside the named ``.cad`` graph:

================  =============================  =======================
function          terminal node identifier       backend
================  =============================  =======================
``cad.fea``       ``com.cad.sim.solver``         CalculiX (linear static)
``cad.impact``    ``com.cad.sim.crash_solver``   OpenCourant (explicit dynamics)
``cad.topopt``    ``com.cad.sim.topopt_voxel``   Density (SIMP) topology
================  =============================  =======================

Inputs are matched against ``NumberNode`` / ``VariableNode`` instances in the
``.cad`` graph whose ``exposed_name`` property equals the kwarg name, and
against named ``CadQueryCodeNode`` or ``FreeCadPartNode`` parameters.  The optional ``_settings``
mapping can also drive validated numeric material, mesh, load, impact,
and topology-optimization properties discovered by the function-block UI. Results
are wrapped in :class:`CadResult`, which gives attribute *and* dict access plus
a small fixed-name standard subset (``max_stress``, ``compliance``, ``mass``,
``volume``, ``peak_disp``, …) so user code is stable across graph versions.

Each call executes the selected solver's upstream dependencies. Studies with
several matching solvers require ``_solver="saved-node-id"`` or a unique name.
Completed evaluations use a process-local LRU cache capped at 32 entries and
256 MiB of estimated Python/array memory. Keys include project and input-file
metadata, analysis kind, solver selection, numeric inputs and settings.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from pylcss.cae.design_studio.core.evaluation_cache import EvaluationCache, result_size_bytes
from pylcss.cae.design_studio.core.graph import node_name, upstream_closure
from pylcss.cae.design_studio.core.parameters import (
    PARAMETER_NAME_RE,
    parse_parameter_text, format_parameter_text, parameter_slot_indices,
)
from pylcss.cae.design_studio.core.study_dependencies import (
    study_dependency_fingerprint as _study_dependency_fingerprint,
)
from pylcss.cae.design_studio.core.study_schema import (
    STUDY_NODE_SCHEMAS, MATERIAL_CUSTOM_PROPERTIES,
    study_node_identifier as _override_identifier,
)

logger = logging.getLogger(__name__)

# Public API surface — kept short on purpose.
__all__ = [
    "fea", "impact", "crash", "topopt", "CadResult", "clear_cache",
    "discover_exposed_parameters",
    "discover_override_controls",
]


# ──────────────────────────────────────────────────────────────────────
# Terminal solver identifiers
# ──────────────────────────────────────────────────────────────────────
_FEA_ID    = "com.cad.sim.solver"
_CRASH_ID  = "com.cad.sim.crash_solver"
_TOPOPT_IDS = ("com.cad.sim.topopt_voxel",)


def discover_override_controls(session_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """List numeric controls using the same metadata as override validation."""
    controls: list[dict[str, Any]] = []
    for node_id, record in (session_data.get("nodes", {}) or {}).items():
        identifier = _override_identifier(record.get("type_", ""))
        schema = STUDY_NODE_SCHEMAS.get(identifier)
        if schema is None:
            continue
        custom = record.get("custom", {}) or {}
        node_name = str(record.get("name") or identifier.rsplit(".", 1)[-1])
        for prop, label in schema.numeric_controls.items():
            value = custom.get(prop)
            if not isinstance(value, (bool, int, float)):
                continue
            controls.append(
                {
                    "key": f"{node_id}::{prop}",
                    "node_id": str(node_id),
                    "node": node_name,
                    "group": schema.group,
                    "property": prop,
                    "label": label,
                    "value": value,
                }
            )
    return controls


def discover_exposed_parameters(session_data: Mapping[str, Any]) -> list[str]:
    """Return every numeric input name accepted by the saved CAD runtime.

    This is shared with the original Function Block editor so its mapping list
    cannot drift from what :func:`fea`, :func:`crash`, and :func:`topopt`
    actually accept.
    """
    names: set[str] = set()
    for node_data in (session_data.get("nodes", {}) or {}).values():
        if not isinstance(node_data, Mapping):
            continue
        node_type = str(node_data.get("type_", "")).lower()
        custom = node_data.get("custom", {}) or {}
        if not isinstance(custom, Mapping):
            continue
        exposed = str(custom.get("exposed_name") or "").strip()
        if not exposed and "variable" in node_type:
            exposed = str(custom.get("variable_name") or "").strip()
        if exposed and PARAMETER_NAME_RE.fullmatch(exposed):
            names.add(exposed)
        for index in parameter_slot_indices(custom):
            parameter = str(custom.get(f"param_{index}_name") or "").strip()
            if PARAMETER_NAME_RE.fullmatch(parameter):
                names.add(parameter)
        raw_parameters = str(custom.get("parameters") or "").strip()
        if raw_parameters:
            try:
                names.update(parse_parameter_text(raw_parameters))
            except (SyntaxError, ValueError):
                logger.debug(
                    "Could not parse saved Code Part parameters while "
                    "discovering the Function Block interface.",
                    exc_info=True,
                )
    return sorted(names)


# ──────────────────────────────────────────────────────────────────────
# Result wrapper
# ──────────────────────────────────────────────────────────────────────
class CadResult:
    """Standardised view of a CAD-graph evaluation result.

    Fields are standardized per analysis kind, with zero defaults for most
    missing numeric values. Optional diagnostics remain optional. The raw result dict from the
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
        """Return a shallow raw-result copy; arrays and mesh objects are shared."""
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
        reaction = np.asarray(
            raw.get("reaction_force", np.zeros(3)), dtype=float
        ).reshape(-1)
        std["reaction_force"] = tuple(float(v) for v in reaction[:3])
        std["reaction_magnitude"] = float(
            raw.get("reaction_magnitude", np.linalg.norm(reaction))
        )

    elif kind in {"impact", "crash"}:
        std["max_stress"]      = float(raw.get("peak_stress", 0.0))
        std["peak_disp"]       = float(raw.get("peak_displacement", 0.0))
        std["absorbed_energy"] = float(raw.get("absorbed_energy", 0.0))
        std["absorbed_energy_kj"] = float(
            raw.get(
                "absorbed_energy_kj",
                float(raw.get("absorbed_energy", 0.0)) / 1.0e6,
            )
        )
        std["n_failed"]        = int(raw.get("n_failed", 0))
        for field in (
            "peak_force",
            "mean_force",
            "crush_force_efficiency",
            "specific_energy_absorption",
            "crush_distance",
            "peak_acceleration_g",
            "delta_v",
        ):
            std[field] = float(raw.get(field, 0.0) or 0.0)
        std["quality_status"] = str(raw.get("quality_status") or "")
        std["numerical_status"] = str(raw.get("numerical_status") or "")
        std["physical_validation_status"] = str(
            raw.get("physical_validation_status") or ""
        )
        std["ml_eligible"] = bool(raw.get("ml_eligible"))
        if raw.get("energy_balance_max_error") is not None:
            std["energy_balance_max_error"] = float(
                raw["energy_balance_max_error"]
            )
        if raw.get("mass_balance_max_error") is not None:
            std["mass_balance_max_error"] = float(
                raw["mass_balance_max_error"]
            )

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
        std["target_vol_frac"] = float(raw.get("target_vol_frac") or 0.0)
        std["compliance"] = float(raw.get("compliance") or 0.0)
        std["mass"] = float(
            raw.get("recovered_design_mass", raw.get("mass")) or 0.0
        )
        std["volume"] = float(
            raw.get("recovered_design_volume", raw.get("volume")) or 0.0
        )
        std["density_equivalent_mass"] = float(
            raw.get("density_equivalent_mass", raw.get("mass")) or 0.0
        )
        std["density_equivalent_volume"] = float(
            raw.get("density_equivalent_volume", raw.get("volume")) or 0.0
        )
        std["total_volume"] = float(raw.get("total_volume") or 0.0)

    return std


# ──────────────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────────────
_cache = EvaluationCache()
# Keep the Python wrapper alive when the runtime creates QApplication itself.
# Without a strong reference, headless ``cad.topopt(...)`` calls can destroy
# the fallback application before NodeGraphQt constructs its first widget.
_headless_qapp: Any | None = None


def clear_cache() -> None:
    """Release every cached evaluation and its retained arrays."""
    _cache.clear()


# ──────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────
def fea(
    cad_path: str,
    _settings: Mapping[str, Any] | None = None,
    *,
    _solver: str | None = None,
    **inputs,
) -> CadResult:
    """Run the static FEA solver; select a saved node id/name with _solver."""
    return _evaluate(cad_path, inputs, _FEA_ID, "fea", _settings, _solver)


def impact(
    cad_path: str,
    _settings: Mapping[str, Any] | None = None,
    *,
    _solver: str | None = None,
    **inputs,
) -> CadResult:
    """Run the selected explicit-impact solver and its upstream dependencies."""
    return _evaluate(cad_path, inputs, _CRASH_ID, "impact", _settings, _solver)


def crash(
    cad_path: str,
    _settings: Mapping[str, Any] | None = None,
    *,
    _solver: str | None = None,
    **inputs,
) -> CadResult:
    """Backward-compatible impact entry point for existing function blocks."""
    return _evaluate(cad_path, inputs, _CRASH_ID, "crash", _settings, _solver)


def topopt(
    cad_path: str,
    _settings: Mapping[str, Any] | None = None,
    *,
    _solver: str | None = None,
    **inputs,
) -> CadResult:
    """Run the selected topology solver and its upstream dependencies."""
    return _evaluate(cad_path, inputs, _TOPOPT_IDS, "topopt", _settings, _solver)


# ──────────────────────────────────────────────────────────────────────
# Core driver
# ──────────────────────────────────────────────────────────────────────
def _evaluate(
    cad_path: str,
    inputs: Mapping[str, Any],
    terminal_id: str | Sequence[str],
    kind: str,
    settings: Mapping[str, Any] | None = None,
    solver_selector: str | None = None,
) -> CadResult:
    abs_path = os.path.abspath(str(cad_path))
    if not os.path.isfile(abs_path):
        # Fallback: resolve repo-relative paths against the PyLCSS repo root
        # so saved models can reference `data/cad_environment/foo.cad` portably.
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
        stat = os.stat(abs_path)
        project_fingerprint = (int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        project_fingerprint = (0, 0)

    canonical_inputs = tuple(sorted((str(k), _to_float(v)) for k, v in inputs.items()))
    canonical_settings = tuple(
        sorted((str(k), _to_float(v)) for k, v in (settings or {}).items())
    )
    cache_key = (
        abs_path,
        project_fingerprint,
        _study_dependency_fingerprint(abs_path),
        kind,
        str(solver_selector) if solver_selector is not None else None,
        canonical_inputs,
        canonical_settings,
    )

    cached = _cache.get(cache_key)
    if cached is not None:
        logger.debug("cad runtime: cache hit %s %s", kind, abs_path)
        return cached

    logger.info(
        "cad runtime: evaluating %s on %s with %d input(s)",
        kind,
        abs_path,
        len(canonical_inputs),
    )

    _ensure_qapp()
    graph = None
    try:
        graph = _load_graph(abs_path)
        terminal = _select_terminal_node(graph, terminal_id, solver_selector)

        _set_count, available_names = _apply_exposed_inputs(
            graph, dict(canonical_inputs)
        )
        requested = {k for k, _ in canonical_inputs}
        if not requested.issubset(available_names):
            missing = sorted(requested - available_names)
            raise KeyError(
                f"CAD graph {abs_path!r} has no exposed parameters named "
                f"{missing}. Available: {sorted(available_names)}"
            )
        _apply_property_overrides(graph, dict(canonical_settings))

        from pylcss.cae.design_studio.engine import execute_graph

        execute_graph(upstream_closure(terminal))

        terminal_result = getattr(terminal, "_last_result", None)
        if terminal_result is None:
            expected = (
                "', '".join(terminal_id)
                if isinstance(terminal_id, (list, tuple))
                else str(terminal_id)
            )
            raise RuntimeError(
                f"CAD graph {abs_path!r} produced no result for terminal node "
                f"'{expected}'. Add the expected solver/optimisation node to "
                "the graph."
            )

        # CadResult copies the result mapping and retains any solver arrays, so
        # the temporary Qt graph can be destroyed immediately after evaluation.
        wrapped = CadResult(kind, terminal_result)
        _cache.put(
            cache_key, wrapped, result_size_bytes((wrapped.raw(), wrapped.standard()))
        )
        return wrapped
    finally:
        _dispose_graph(graph)


# ──────────────────────────────────────────────────────────────────────
# Graph helpers
# ──────────────────────────────────────────────────────────────────────
def _load_graph(abs_path: str):
    """Prepare a saved session using the same migrations as the GUI loader."""
    from NodeGraphQt import NodeGraph
    from pylcss.cae.design_studio.node_library import NODE_CLASS_MAPPING
    from pylcss.cae.design_studio.session_persistence import (
        parse_design_studio_session,
        prepare_design_studio_session,
    )

    with open(abs_path, "r", encoding="utf-8") as handle:
        session_data = parse_design_studio_session(handle.read())
    graph = NodeGraph()
    try:
        registered = {}
        for node_class in dict.fromkeys(NODE_CLASS_MAPPING.values()):
            try:
                graph.register_node(node_class)
                registered[node_class.__identifier__] = node_class
            except Exception as exc:
                logger.warning(
                    "Could not register CAD node %s while loading %s: %s",
                    node_class.__name__,
                    abs_path,
                    exc,
                )
        omitted = prepare_design_studio_session(session_data, known_classes=registered)
        if omitted:
            logger.warning("Omitted unsupported study nodes: %s", ", ".join(omitted))
        graph.deserialize_session(session_data)
        loaded_nodes = list(graph.all_nodes())
        for node in loaded_nodes:
            node._project_dir = str(Path(abs_path).resolve().parent)

        # Preserve saved IDs even when NodeGraphQt assigns new runtime IDs.
        unmatched = set(range(len(loaded_nodes)))
        for saved_index, (saved_id, saved_data) in enumerate(
            session_data["nodes"].items()
        ):
            saved_name = str(saved_data.get("name") or "")
            matches = [
                idx for idx in unmatched if str(loaded_nodes[idx].name()) == saved_name
            ]
            if len(matches) == 1:
                loaded_index = matches[0]
            elif saved_index in unmatched:
                loaded_index = saved_index
            else:
                continue
            loaded_nodes[loaded_index]._pylcss_saved_node_id = str(saved_id)
            unmatched.discard(loaded_index)
        return graph
    except Exception:
        _dispose_graph(graph)
        raise


def _dispose_graph(graph) -> None:
    """Destroy a temporary NodeGraph while QApplication is still alive."""
    if graph is None:
        return
    try:
        graph.clear_session()
    except Exception:
        logger.debug("cad runtime: graph session cleanup failed", exc_info=True)
    try:
        widget = graph.widget
        widget.close()
        widget.deleteLater()
    except Exception:
        logger.debug("cad runtime: graph widget cleanup failed", exc_info=True)
    try:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        logger.debug("cad runtime: Qt cleanup events failed", exc_info=True)


def _apply_exposed_inputs(graph, inputs: Mapping[str, float]) -> tuple[int, set]:
    """Push kwargs into named CAD graph parameters.

    Supported targets:
    - NumberNode / VariableNode via ``exposed_name`` (or VariableNode name).
    - CadQueryCodeNode / FreeCadPartNode named parameter slots.
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
    """Apply validated ``node_id::property`` overrides to a fresh CAD graph.

    Legacy ``node_name::property`` keys remain accepted when the display name
    uniquely identifies one node.
    """
    if not settings:
        return 0

    nodes_by_name = {}
    nodes_by_id = {}
    for node in graph.all_nodes():
        node_name = node.name() if hasattr(node, "name") else ""
        nodes_by_name.setdefault(str(node_name), []).append(node)
        nodes_by_id[str(getattr(node, "id", ""))] = node
        saved_node_id = getattr(node, "_pylcss_saved_node_id", None)
        if saved_node_id:
            nodes_by_id[str(saved_node_id)] = node

    applied = 0
    for key, numeric_value in settings.items():
        if "::" not in key:
            raise KeyError(
                f"Invalid Design Studio setting key {key!r}; expected 'node_id::property'."
            )
        node_key, prop = key.rsplit("::", 1)
        node = nodes_by_id.get(node_key)
        if node is None:
            named = nodes_by_name.get(node_key, [])
            if len(named) > 1:
                raise KeyError(
                    f"Design Studio node name {node_key!r} is ambiguous; refresh the "
                    "coupling so it uses stable node IDs."
                )
            node = named[0] if named else None
        if node is None:
            raise KeyError(
                f"Design Studio node {node_key!r} no longer exists in the saved study."
            )

        identifier = _override_identifier(getattr(node, "__identifier__", ""))
        schema = STUDY_NODE_SCHEMAS.get(identifier)
        allowed = schema.numeric_controls if schema is not None else {}
        if prop not in allowed or not node.has_property(prop):
            node_name = node.name() if hasattr(node, "name") else node_key
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
        # Named material presets own their database values.  A numeric override
        # must switch the affected material to Custom or the backend would
        # silently ignore the requested value.
        if identifier == "com.cad.sim.material" and prop in MATERIAL_CUSTOM_PROPERTIES:
            node.set_property("preset", "Custom")
        node.set_property(prop, value)
        _mark_node_dirty(node)
        applied += 1
    return applied


def _is_code_part_node(node) -> bool:
    return (
        getattr(node, "__identifier__", "") in {
            "com.cad.code_part", "com.cad.freecad_part",
        }
        or node.__class__.__name__ in {"CadQueryCodeNode", "FreeCadPartNode"}
    )


def _mark_node_dirty(node) -> None:
    # Force re-execution: bust the engine's per-node dirty-state cache.
    node._last_result = None
    node._last_input_hash = None
    node._dirty = True
    node._force_execute = True


def _apply_code_part_inputs(node, inputs: Mapping[str, float], available: set) -> int:
    applied = 0

    count_getter = getattr(node, "get_param_count", None)
    if callable(count_getter):
        indices = range(1, count_getter() + 1)
    else:
        properties = getattr(node, "properties", None)
        custom = properties().get("custom", {}) if callable(properties) else {}
        indices = parameter_slot_indices(custom)
    for idx in indices:
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
            logger.warning(
                "cad runtime: failed to set code parameter %s: %s", pname, exc
            )
        else:
            _mark_node_dirty(node)
            if getattr(node, "__identifier__", "") == "com.cad.freecad_part":
                node._parameter_override_pending = True
            applied += 1

    if not node.has_property("parameters"):
        return applied

    try:
        params = parse_parameter_text(node.get_property("parameters") or "")
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
        node.set_property("parameters", format_parameter_text(params))
        _mark_node_dirty(node)

    return applied


def _select_terminal_node(graph, terminal_id: str | Sequence[str], selector=None):
    """Require explicit selection when a study contains several matching solvers."""
    terminal_ids = {terminal_id} if isinstance(terminal_id, str) else set(terminal_id)
    candidates = [
        node
        for node in graph.all_nodes()
        if getattr(node, "__identifier__", "") in terminal_ids
    ]
    if selector is not None:
        key = str(selector).strip()
        candidates = [
            node
            for node in candidates
            if key
            in {
                str(getattr(node, "id", "")),
                str(getattr(node, "_pylcss_saved_node_id", "")),
                node_name(node),
            }
        ]
        if not candidates:
            raise KeyError(f"No matching solver {selector!r} in the saved study.")
    if not candidates:
        raise RuntimeError(
            f"Saved study has no solver with identifier {sorted(terminal_ids)}."
        )
    if len(candidates) > 1:
        choices = [
            f"{node_name(node)} ({getattr(node, '_pylcss_saved_node_id', getattr(node, 'id', ''))})"
            for node in candidates
        ]
        raise ValueError(
            "Several matching solvers are available: "
            + ", ".join(choices)
            + ". Choose a saved node ID or unique name with _solver."
        )
    return candidates[0]


# ──────────────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────────────
def _ensure_qapp():
    """NodeGraphQt requires a QApplication. The sysmod GUI already has one; this
    is a defensive fallback for headless test/script contexts."""
    global _headless_qapp
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
    _headless_qapp = app
    return app


def _to_float(value: Any) -> float:
    """Coerce kwarg values so cache keys are hashable & stable."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Design Studio inputs and settings must be numeric, got {value!r}.") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"Design Studio inputs and settings must be finite, got {value!r}.")
    return numeric
