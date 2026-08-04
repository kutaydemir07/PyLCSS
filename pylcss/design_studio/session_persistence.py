# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Persistence policy for compact, forward-compatible Design Studio sessions."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


from pylcss.design_studio.topology_optimization.integration.study_identity import (
    LATTICE_SOLVER_CLASS_NAME,
    LATTICE_SOLVER_IDENTIFIER,
    TOPOLOGY_SOLVER_CLASS_NAME,
    is_density_study_record,
    is_lattice_study_record,
)

# Guided studies store engineering intent.  Numerical controls derived from
# that intent (grid dimensions, optimizer, tolerance, SIMP penalty, recovery
# parameters, and iteration budgets) belong in the result report/sidecar.
_GUIDED_TOPOLOGY_BASE_PROPERTIES = frozenset(
    {
        "workflow_mode",
        "design_goal",
        "manufacturing_process",
        "volfrac",
        "minimum_member_size_mm",
        "exclusion_scope",
        "exclusion_thickness_mode",
        "structure_mode",
        "lattice_settings_mode",
        "visualization",
        "validate_after_optimize",
        "cad_export_filename",
        "description",
        "tags",
        "notes",
    }
)

# A lattice's physical dimensions are printer capabilities, not numerics
# derived from the design goal, so they are intent and must survive a save.
# These names have to match the properties `LatticeOptVoxelNode` actually
# declares: they are looked up by name and anything that does not match is
# dropped silently, which is how a Guided lattice study used to be re-opened
# with its cell pitch, wall thickness and mass budget reset to zero.
_LATTICE_INTENT_PROPERTIES = frozenset(
    {
        "lattice_cell_size_mm",
        "lattice_member_thickness_mm",
        "lattice_skin_thickness_mm",
        # The mass budget for the manufactured lattice. Cell pitch and member
        # thickness do not predict it, so losing it loses the one number that
        # says what the part weighs.
        "lattice_target_relative_density",
        "lattice_porosity",
    }
)

# Every property that moved to `LatticeOptVoxelNode` when lattice optimization
# became its own node. A study saved before the split stores these on a
# topology record, where they are now unknown names; NodeGraphQt rejects an
# unknown property outright, so a record that is not retyped has to be cleaned.
# Kept as literals because this module runs before deserialization and must not
# import the node classes to read their declarations.
_LATTICE_ONLY_PROPERTIES = frozenset(
    _LATTICE_INTENT_PROPERTIES
    | {
        "lattice_settings_mode",
        "structure_cell_size_voxels",
        "structure_member_thickness_voxels",
        "structure_skin_thickness_voxels",
        "lattice_variable_density",
        "lattice_min_relative_density",
        "lattice_max_relative_density",
        "lattice_solid_transition_density",
        "optimize_lattice_members",
        "lattice_max_member_thickness_voxels",
        "lattice_member_sizing_iterations",
        "lattice_buckling_length_factor",
    }
)


# Property names withdrawn from the study nodes. NodeGraphQt raises on an
# unknown custom property rather than ignoring it, so every saved project that
# still carries one of these fails to open until it is stripped here.
_RETIRED_TOPOLOGY_PROPERTIES = frozenset(
    {
        # Retired fidelity dropdowns, replaced by physical feature sizes.
        "quality_preset",
        "analysis_resolution",
        # The robust dilated/nominal/eroded formulation and its thresholds.
        "robust_enabled",
        "robust_eta_offset",
        "robust_eta_dilated",
        "robust_eta_eroded",
        "robust_power",
    }
)


def parse_design_studio_session(text: str) -> dict[str, Any]:
    """Parse one unambiguous Design Studio JSON object.

    Standard ``json.loads`` silently keeps the last value of a duplicate key,
    so a malformed study containing two ``tol`` fields appears valid while its
    executed setting depends on file order. Engineering inputs must be
    unambiguous; reject duplicates and let the user repair or re-save the file.
    """

    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    "Design Studio JSON contains duplicate field "
                    f"{key!r}; every engineering setting must occur once."
                )
            result[key] = value
        return result

    try:
        session = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Design Studio file is not valid JSON: {exc}") from exc
    if not isinstance(session, dict) or not isinstance(
        session.get("nodes"), dict
    ):
        raise ValueError(
            "Design Studio file must contain a JSON object with a nodes object."
        )
    return session


def _is_topology_node(record: Any) -> bool:
    return is_density_study_record(record)


def _guided_topology_property_names(
    custom: dict[str, Any],
    *,
    lattice_study: bool = False,
) -> set[str]:
    """Return the context-sensitive intent fields for one Guided study."""
    names = set(_GUIDED_TOPOLOGY_BASE_PROPERTIES)
    process = str(custom.get("manufacturing_process") or "None").strip().lower()
    structure = str(custom.get("structure_mode") or "Solid Envelope").strip().lower()
    goal = str(custom.get("design_goal") or "Lightweight Stiffness").strip().lower()
    exclusion_mode = str(
        custom.get("exclusion_thickness_mode") or "Program Controlled"
    ).strip().lower()

    try:
        maximum_member_size = float(
            custom.get("maximum_member_size_mm") or 0.0
        )
    except (TypeError, ValueError):
        maximum_member_size = 0.0
    if maximum_member_size > 0.0:
        names.add("maximum_member_size_mm")
    if process in {"additive", "additive + symmetric"}:
        names.update({"overhang_build_axis", "overhang_angle_deg"})
    if process == "cast / moulded":
        names.add("pull_out_axis")
    if process == "extruded":
        names.add("extrusion")
    if process in {"symmetric", "additive + symmetric"}:
        names.add("symmetry")
    if "stress" in goal:
        names.add("yield_stress")
    if exclusion_mode == "manual":
        names.add("exclusion_thickness_mm")
    # The lattice node always owns a cell, so its manufacturing dimensions are
    # always intent. The structure check remains for topology records saved
    # before lattice optimization became its own node.
    if lattice_study or structure != "solid envelope":
        names.update(_LATTICE_INTENT_PROPERTIES)
    return names


def migrate_lattice_topology_nodes(session: dict[str, Any]) -> dict[str, Any]:
    """Retype pre-split lattice studies onto the lattice optimization node.

    Lattice optimization used to be an output mode of the topology node. It is
    now its own node, because it is its own design problem: the graded density
    field it optimizes is the very thing a solid study penalizes away. A saved
    study that selected a cell family therefore describes a lattice study and
    is loaded as one, keeping its family, pitch, wall thickness and mass
    budget. A study that selected the solid envelope simply loses the setting,
    which no longer exists on the topology node and would otherwise abort the
    load as an unknown property.

    Mutates *session* in place and returns it, matching the other migrations
    that run before NodeGraphQt deserialization.
    """
    from pylcss.design_studio.topology_optimization.manufacturing import (
        normalize_family_key,
    )

    nodes = session.get("nodes")
    if not isinstance(nodes, dict):
        return session

    for record in nodes.values():
        if not isinstance(record, dict):
            continue
        type_name = str(record.get("type_") or "")
        if not type_name.endswith(f".{TOPOLOGY_SOLVER_CLASS_NAME}"):
            continue
        custom = record.get("custom")
        custom = custom if isinstance(custom, dict) else {}
        family = normalize_family_key(custom.get("structure_mode"))
        if family and family != "solid":
            record["type_"] = (
                f"{LATTICE_SOLVER_IDENTIFIER}.{LATTICE_SOLVER_CLASS_NAME}"
            )
            continue
        for name in ({"structure_mode"} | _LATTICE_ONLY_PROPERTIES):
            custom.pop(name, None)
    return session


def compact_design_studio_session(session: dict[str, Any]) -> dict[str, Any]:
    """Return a save-ready session with derived Guided TopOpt controls removed.

    Expert nodes are intentionally untouched: an Expert study owns its exact
    numerical controls and must persist them.  Missing Guided properties are
    restored from the node's current policy defaults during deserialization,
    then deterministically resolved from the saved intent at execution time.
    """
    compact = deepcopy(session)
    nodes = compact.get("nodes")
    if not isinstance(nodes, dict):
        return compact

    compacted_count = 0
    for record in nodes.values():
        if not _is_topology_node(record):
            continue
        custom = record.get("custom")
        if not isinstance(custom, dict):
            continue
        workflow = str(custom.get("workflow_mode") or "Guided").strip().lower()
        if workflow != "guided":
            continue
        keep = _guided_topology_property_names(
            custom,
            lattice_study=is_lattice_study_record(record),
        )
        record["custom"] = {
            key: value for key, value in custom.items() if key in keep
        }
        compacted_count += 1

    if compacted_count:
        compact["_guided_topology_policy"] = {
            "version": 1,
            "nodes": compacted_count,
            "storage": "engineering-intent",
        }
    return compact


def expand_guided_topology_session(session: dict[str, Any]) -> dict[str, Any]:
    """Hydrate derived Guided controls before NodeGraphQt deserialization.

    This preserves the compact file format without making execution depend on
    whatever low-level defaults happen to be declared in the node constructor.
    A policy upgrade changes one deterministic mapping, while the saved design
    goal, fidelity, process, physical sizes, and user-visible choices remain
    authoritative.
    """
    nodes = session.get("nodes")
    if not isinstance(nodes, dict):
        return session

    from pylcss.design_studio.topology_optimization.configuration.presets import (
        industrial_lattice_defaults,
        industrial_topopt_defaults,
    )

    for record in nodes.values():
        if not _is_topology_node(record):
            continue
        custom = record.get("custom")
        if not isinstance(custom, dict):
            continue
        # Remove retired/legacy property keys that are no longer registered
        # NodeGraphQt properties. An unknown name is not ignored on load — it
        # raises and takes the whole project with it — so a property that is
        # withdrawn has to be listed here in the same change that withdraws it.
        for retired in _RETIRED_TOPOLOGY_PROPERTIES:
            custom.pop(retired, None)
        workflow = str(custom.get("workflow_mode") or "Guided").strip().lower()
        if workflow != "guided":
            continue
        if is_lattice_study_record(record):
            resolved = industrial_lattice_defaults(
                custom.get("design_goal"),
                custom.get("manufacturing_process"),
                structure_mode=custom.get("structure_mode"),
            )
        else:
            resolved = industrial_topopt_defaults(
                custom.get("design_goal"),
                custom.get("manufacturing_process"),
            )
        record["custom"] = {**resolved, **custom}
    return session
