# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Headless CAD-graph evaluator exposed to system-modeling function blocks.

The compiled code of a sysmod ``CustomBlockNode`` sees this module bound to the
name ``cad`` (see :mod:`pylcss.system_modeling.compiler`).  A function
block can then write::

    r = cad.fea("front_panel.cad", thickness=t, fillet_r=ro)
    return r.max_stress, r.mass

Three primary entry points are provided, each one targeting a different terminal solver
node inside the named ``.cad`` graph:

================  =============================  =======================
function          terminal node identifier       backend
================  =============================  =======================
``cad.fea``       ``com.cad.sim.solver``         CalculiX (linear static)
``cad.impact``    ``com.cad.sim.crash_solver``   OpenRadioss (explicit dynamics)
``cad.topopt``    ``com.cad.sim.topopt_voxel``   Density / level-set topology
                  ``com.cad.sim.lattice_voxel``  Variable-density lattice
================  =============================  =======================

Inputs are matched against ``NumberNode`` / ``VariableNode`` instances in the
``.cad`` graph whose ``exposed_name`` property equals the kwarg name, and
against named ``CadQueryCodeNode`` or ``FreeCadPartNode`` parameters.  The optional ``_settings``
mapping can also drive validated numeric material, mesh, load, impact,
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
from typing import Any, Dict, Mapping, Sequence

import numpy as np

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
_TOPOPT_IDS = ("com.cad.sim.topopt_voxel", "com.cad.sim.lattice_voxel")


# Numeric Design Studio controls that are safe and meaningful to drive from a
# system-model function block.  File paths, display options, face selections,
# and backend switches intentionally remain owned by the saved .cad study.
_OVERRIDEABLE_PROPERTIES = {
    "com.cad.sim.material": (
        "youngs_modulus", "poissons_ratio", "density",
        "thermal_conductivity", "yield_strength", "tangent_modulus",
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
        "force_x", "force_y", "force_z", "gravity_accel",
    ),
    "com.cad.topopt.load": ("force_x", "force_y", "force_z"),
    "com.cad.sim.pressure_load": ("pressure",),
    "com.cad.sim.impact": (
        "velocity_x", "velocity_y", "velocity_z", "node_tolerance",
        "wall_friction", "wall_gap_mm",
    ),
    "com.cad.sim.crash_solver": (
        "end_time", "n_frames", "time_steps", "enable_mass_scaling",
        "impactor_mass_kg",
    ),
    "com.cad.sim.topopt_voxel": (
        "nelx", "nely", "nelz", "volfrac", "rmin", "penal",
        "density_cutoff", "max_iter", "tol", "convergence_patience",
        "stress_constraint", "yield_stress", "max_member_size_voxels",
        "maximum_member_size_mm",
        "minimum_member_size_mm", "minimum_void_size_mm",
        "overhang_angle_deg",
        "topology_convergence_enabled", "topology_convergence_levels",
        "exclusion_thickness_mm",
        "pattern_repeat",
    ),
}
# The lattice study drives the same solver, so it exposes the topology
# controls plus the manufacturing dimensions of its cell.
_OVERRIDEABLE_PROPERTIES["com.cad.sim.lattice_voxel"] = (
    *_OVERRIDEABLE_PROPERTIES["com.cad.sim.topopt_voxel"],
    "structure_cell_size_voxels", "structure_member_thickness_voxels",
    "structure_skin_thickness_voxels",
    "lattice_cell_size_mm", "lattice_member_thickness_mm",
    "lattice_skin_thickness_mm", "lattice_target_relative_density",
    "lattice_variable_density",
    "lattice_min_relative_density", "lattice_max_relative_density",
    "lattice_solid_transition_density", "lattice_porosity",
)

_OVERRIDE_GROUPS = {
    "com.cad.sim.material": "Material",
    "com.cad.sim.mesh": "Mesh",
    "com.cad.sim.constraint": "Boundary condition",
    "com.cad.sim.load": "Load",
    "com.cad.topopt.load": "Topology load",
    "com.cad.sim.pressure_load": "Load",
    "com.cad.sim.impact": "Impact",
    "com.cad.sim.crash_solver": "Impact solver",
    "com.cad.sim.topopt_voxel": "Topology optimization",
    "com.cad.sim.lattice_voxel": "Lattice optimization",
}

_PROPERTY_LABELS = {
    "youngs_modulus": "Young's modulus",
    "poissons_ratio": "Poisson's ratio",
    "density": "Density",
    "thermal_conductivity": "Thermal conductivity",
    "yield_strength": "Yield strength",
    "tangent_modulus": "Tangent modulus",
    "failure_strain": "Failure strain",
    "force_x": "Force X",
    "force_y": "Force Y",
    "force_z": "Force Z",
    "relative_stiffness": "Relative joint stiffness",
    "total_heat": "Total heat input",
    "weight": "Case weight",
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
    "structure_cell_size_voxels": "Lattice cell size",
    "structure_member_thickness_voxels": "Minimum lattice wall/member",
    "lattice_cell_size_mm": "Lattice cell pitch",
    "lattice_member_thickness_mm": "Lattice wall/member thickness",
    "lattice_skin_thickness_mm": "Lattice skin thickness",
    "lattice_target_relative_density": "Lattice target relative density",
    "lattice_min_relative_density": "Minimum lattice relative density",
    "lattice_max_relative_density": "Maximum lattice relative density",
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
                # NodeGraphQt IDs are stable in a saved session and avoid the
                # ambiguity of several default-named Material or Load nodes.
                "key": f"{node_id}::{prop}",
                "node_id": str(node_id),
                "node": node_name,
                "group": _OVERRIDE_GROUPS.get(identifier, "Analysis setting"),
                "property": prop,
                "label": _PROPERTY_LABELS.get(prop, prop.replace("_", " ").title()),
                "value": value,
            })
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
        if exposed and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", exposed):
            names.add(exposed)
        max_params = 8 if "freecad_part" in node_type else 6
        for index in range(1, max_params + 1):
            parameter = str(custom.get(f"param_{index}_name") or "").strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", parameter):
                names.add(parameter)
        raw_parameters = str(custom.get("parameters") or "").strip()
        if raw_parameters:
            try:
                names.update(_parse_code_part_parameters(raw_parameters))
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
        std["thermal_compliance"] = float(
            raw.get("thermal_compliance") or 0.0
        )
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
_cache: Dict[tuple, CadResult] = {}
_cache_lock = threading.Lock()
# Keep the Python wrapper alive when the runtime creates QApplication itself.
# Without a strong reference, headless ``cad.topopt(...)`` calls can destroy
# the fallback application before NodeGraphQt constructs its first widget.
_headless_qapp: Any | None = None


def clear_cache() -> None:
    """Drop every cached CAD-graph evaluation (per-process)."""
    with _cache_lock:
        _cache.clear()


_DEPENDENCY_SUFFIXES = {
    ".step", ".stp", ".iges", ".igs", ".stl", ".obj", ".fcstd",
    ".brep", ".k", ".rad",
}


def _study_dependency_fingerprint(cad_path: str) -> tuple:
    """Fingerprint geometry/deck files referenced by a saved study.

    The runtime cache previously watched only the ``.cad`` JSON timestamp, so
    editing an imported STEP/STL/FreeCAD model or a referenced Radioss deck
    could return an old solve. Missing paths are included too, which means the
    cache also invalidates when a previously missing dependency appears.
    """
    project = Path(cad_path).resolve()
    try:
        session = json.loads(project.read_text(encoding="utf-8"))
    except Exception:
        return ()

    try:
        from pylcss.config import BASE_DIR
        repo_root = Path(BASE_DIR).resolve().parent
    except Exception:
        repo_root = project.parent

    records = []
    for node_data in (session.get("nodes", {}) or {}).values():
        node_type = str(node_data.get("type_", "")).lower()
        custom = node_data.get("custom", {}) or {}
        for prop, raw_value in custom.items():
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            raw = os.path.expandvars(os.path.expanduser(raw_value.strip()))
            suffix = Path(raw).suffix.lower()
            if suffix not in _DEPENDENCY_SUFFIXES:
                continue

            candidates = []
            if "freecad_part" in node_type and str(prop) == "fcstd_filename":
                try:
                    from pylcss.design_studio.freecad_bridge.paths import freecad_data_dir
                    candidates.append(freecad_data_dir(create=False) / Path(raw).name)
                except Exception:
                    pass
            raw_path = Path(raw)
            if raw_path.is_absolute():
                candidates.append(raw_path)
            else:
                candidates.extend((project.parent / raw_path, repo_root / raw_path))

            resolved = next((p.resolve() for p in candidates if p.is_file()), None)
            if resolved is None:
                records.append((str(prop), raw, "missing"))
                continue
            try:
                stat = resolved.stat()
                records.append((
                    str(resolved), int(stat.st_mtime_ns), int(stat.st_size),
                ))
            except OSError:
                records.append((str(resolved), "unreadable"))
    return tuple(sorted(set(records), key=repr))


# ──────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────
def fea(cad_path: str, _settings: Mapping[str, Any] | None = None, **inputs) -> CadResult:
    """Run the FEA-solver path of a CAD graph and return its scalar results."""
    return _evaluate(cad_path, inputs, terminal_id=_FEA_ID, kind="fea", settings=_settings)


def impact(cad_path: str, _settings: Mapping[str, Any] | None = None, **inputs) -> CadResult:
    """Run the explicit-impact path of a CAD graph and return its scalar results."""
    return _evaluate(
        cad_path,
        inputs,
        terminal_id=_CRASH_ID,
        kind="impact",
        settings=_settings,
    )


def crash(cad_path: str, _settings: Mapping[str, Any] | None = None, **inputs) -> CadResult:
    """Backward-compatible name for existing models; prefer :func:`impact`."""
    return _evaluate(
        cad_path,
        inputs,
        terminal_id=_CRASH_ID,
        kind="crash",
        settings=_settings,
    )


def topopt(cad_path: str, _settings: Mapping[str, Any] | None = None, **inputs) -> CadResult:
    """Run the saved density or level-set topology study in a CAD graph."""
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
        canonical_inputs,
        canonical_settings,
    )

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
    graph = None
    try:
        graph = _load_graph(abs_path)

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
                f"'{expected}'. Add the expected solver/optimisation node to "
                "the graph."
            )

        # CadResult copies the result mapping and retains any solver arrays, so
        # the temporary Qt graph can be destroyed immediately after evaluation.
        wrapped = CadResult(kind, terminal_result)
        with _cache_lock:
            _cache[cache_key] = wrapped
        return wrapped
    finally:
        _dispose_graph(graph)


# ──────────────────────────────────────────────────────────────────────
# Graph helpers
# ──────────────────────────────────────────────────────────────────────
def _load_graph(abs_path: str):
    """Spin up a fresh ``NodeGraph``, register every CAD node, deserialise the file."""
    from NodeGraphQt import NodeGraph
    from pylcss.design_studio.node_library import NODE_CLASS_MAPPING

    graph = NodeGraph()
    for node_class in dict.fromkeys(NODE_CLASS_MAPPING.values()):
        try:
            graph.register_node(node_class)
        except Exception as exc:
            logger.warning(
                "Could not register CAD node %s while loading %s: %s",
                getattr(node_class, "__name__", node_class), abs_path, exc,
            )

    from pylcss.design_studio.session_persistence import (
        parse_design_studio_session,
    )

    with open(abs_path, "r", encoding="utf-8") as f:
        session_data = parse_design_studio_session(f.read())
    from pylcss.design_studio.crash.conditions import (
        migrate_impact_scenario_properties,
    )
    from pylcss.design_studio.fem.mesh import migrate_removed_mesher_properties

    migrate_impact_scenario_properties(session_data)
    migrate_removed_mesher_properties(session_data)
    from pylcss.design_studio.session_persistence import (
        expand_guided_topology_session,
        migrate_lattice_topology_nodes,
    )

    # Retype before expansion: which guided defaults a study is hydrated with
    # depends on whether it is a lattice study.
    migrate_lattice_topology_nodes(session_data)
    expand_guided_topology_session(session_data)
    graph.clear_session()
    graph.deserialize_session(session_data)
    project_dir = str(Path(abs_path).resolve().parent)
    loaded_nodes = list(graph.all_nodes())
    for node in loaded_nodes:
        node._project_dir = project_dir

    # Hand-authored example files use readable saved keys (for example
    # ``force`` and ``topopt``). NodeGraphQt assigns fresh pointer-like runtime
    # ids when deserialising those keys, so keep the project key on each node.
    # Function-block and AI overrides must remain stable across every load.
    saved_nodes = list((session_data.get("nodes", {}) or {}).items())
    unmatched = set(range(len(loaded_nodes)))
    for saved_index, (saved_id, saved_data) in enumerate(saved_nodes):
        saved_name = str(saved_data.get("name") or "")
        matches = [
            idx for idx in unmatched
            if str(loaded_nodes[idx].name()) == saved_name
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
            raise KeyError(f"Design Studio node {node_key!r} no longer exists in the saved study.")

        identifier = _override_identifier(getattr(node, "__identifier__", ""))
        allowed = _OVERRIDEABLE_PROPERTIES.get(identifier or "", ())
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
        if identifier == "com.cad.sim.material" and prop in {
            "youngs_modulus", "poissons_ratio", "density",
            "thermal_conductivity",
        }:
            node.set_property("preset", "Custom")
        elif identifier == "com.cad.sim.crash_material" and prop in {
            "youngs_modulus", "poissons_ratio", "density", "yield_strength",
            "tangent_modulus", "failure_strain",
        }:
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

    max_params = max(1, int(getattr(node, "MAX_PARAMS", 6) or 6))
    for idx in range(1, max_params + 1):
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
            if getattr(node, "__identifier__", "") == "com.cad.freecad_part":
                node._parameter_override_pending = True
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

    def _depth(node, visited=None) -> int:
        visited = set(visited or ())
        if id(node) in visited:
            return 0
        visited.add(id(node))
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
                count += _depth(up, visited) if up is not node else 0
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
