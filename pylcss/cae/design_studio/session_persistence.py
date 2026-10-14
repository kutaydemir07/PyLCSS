# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Persistence policy for compact, forward-compatible Design Studio sessions."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


from pylcss.cae.design_studio.topology_optimization.integration.study_identity import (
    TOPOLOGY_SOLVER_IDENTIFIER,
    TOPOLOGY_SOLVER_CLASS_NAME,
    is_density_study_record,
)

# Guided studies store engineering intent.  Numerical controls derived from
# that intent (grid dimensions, optimizer, tolerance, SIMP penalty, recovery
# parameters, and iteration budgets) belong in the result report/sidecar.
_GUIDED_TOPOLOGY_BASE_PROPERTIES = frozenset(
    {
        "workflow_mode",
        "design_goal",
        "volfrac",
        "stress_constraint",
        "yield_stress",
        "load_aggregation",
        "structure_mode",
        "lattice_settings_mode",
        "visualization",
        "validate_after_optimize",
        "cad_export_filename",
        "cad_export_format",
        "description",
        "tags",
        "notes",
    }
)

# A lattice's physical dimensions are printer capabilities, not numerics
# derived from the design goal, so they are intent and must survive a save.
# These names have to match the properties the infill node actually
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
        "repair_lattice_connections",
    }
)

# Retired cell controls translated or removed before TopOpt deserialization.
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
        "draw_direction",
        "extrusion_axis",
        "manufacturing_process",
        "max_member_size_mm",
        "max_member_size_voxels",
        "maximum_member_size_mm",
        "minimum_member_size_mm",
        "minimum_void_size_mm",
        "overhang_angle_deg",
        "overhang_direction",
        "pattern_axis",
        "pattern_repeat",
        "symmetry_plane",
        # Automatic interface preservation was replaced by connected CAD
        # non-design and obstacle bodies. Ignore it in older projects.
        "exclusion_scope",
        "exclusion_thickness_mode",
        "exclusion_thickness_mm",
        "load_case_mode",
        # Retired fidelity dropdowns, replaced by physical feature sizes.
        "quality_preset",
        "analysis_resolution",
        # Retired extrusion property.
        "extrusion",
        # Retired manufacturing process and constraints.
        "symmetry",
        "overhang_build_axis",
        "pull_out_axis",
        # Retired mesh smoothing and artificial recovery choices.
        "print_ready_mesh",
        "mesh_decimate_ratio",
        "surface_recovery_method",
        "surface_quality",
        # Retired lattice voxel option; current reconstruction resolves solid
        # occupancy directly and the node no longer registers this property.
        "adaptive_subcell_solid",
        # The robust dilated/nominal/eroded formulation and its thresholds.
        "robust_enabled",
        "robust_eta_offset",
        "robust_eta_dilated",
        "robust_eta_eroded",
        "robust_power",
        # TopOpt now exposes and solves structural studies only.
        "thermal_conductivity",
        "thermal_conductivity_min",
        "thermal_penal",
        "convection_coefficient",
        "structural_weight",
        "thermal_weight",
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
    structure = str(custom.get("structure_mode") or "Solid Envelope").strip().lower()
    # The lattice node always owns a cell, so its manufacturing dimensions are
    # always intent. The structure check remains for topology records saved
    # before lattice generation became a post-processing action.
    if lattice_study or structure != "solid envelope":
        names.update(_LATTICE_INTENT_PROPERTIES)
        # Manual reconstruction settings are user choices even when the
        # analysis grid uses Guided sizing. Saving must not reset them.
        if str(custom.get("lattice_settings_mode") or "Guided").lower() == "manual":
            names.update(_LATTICE_ONLY_PROPERTIES)
    return names


def migrate_cad_reconstruction_properties(session: dict[str, Any]) -> dict[str, Any]:
    """Normalize retired reconstruction selections before graph deserialization."""
    # Structure capability lookup does not load the CAD kernel.
    names = {
        "auto (amrto)": "Auto", "amrto": "AMRTO",
        "amrto bicubic nurbs": "AMRTO", "faceted": "Auto",
        "recovered shape": "Auto", "smooth freeform": "Smooth spline",
    }
    nodes = session.get("nodes")
    if isinstance(nodes, dict):
        for record in nodes.values():
            if not isinstance(record, dict):
                continue
            custom = record.get("custom")
            if isinstance(custom, dict):
                custom.pop('cad_reconstruction_method', None)
                density_node = any(name in str(record.get('type_', '')) for name in
                                   ('topopt_voxel', 'lattice_voxel', 'lattice_infill'))
                if density_node and 'visualization' in custom and custom['visualization'] not in {'CAD', 'Density'}:
                    custom['visualization'] = 'CAD'
                from .topology_optimization.geometry.lattice_cad import topopt_export_formats
                formats = topopt_export_formats(custom.get("structure_mode"))
                if custom.get("cad_export_format") and custom["cad_export_format"].upper() not in formats:
                    custom["cad_export_format"] = formats[0]
                    filename = custom.get("cad_export_filename")
                    if filename:
                        from pathlib import Path
                        custom["cad_export_filename"] = str(Path(filename).with_suffix(".stl"))
                if "STEP" not in formats:
                    custom["generate_cad_after_optimize"] = False
            if isinstance(custom, dict) and "cad_reconstruction_method" in custom:
                key = str(custom["cad_reconstruction_method"]).strip().lower()
                if key in names:
                    custom["cad_reconstruction_method"] = names[key]
    return session


def migrate_lattice_topology_nodes(session: dict[str, Any]) -> dict[str, Any]:
    """Retype retired cellular solvers as TopOpt with post-processing choices.

    Keep node ids, boundary wires and physical cell dimensions. Old solver
    results remain viewable, but the next run solves a solid topology.
    """
    from .topology_optimization.manufacturing import normalize_family_key

    for record in (session.get("nodes") or {}).values():
        if not isinstance(record, dict):
            continue
        type_name = str(record.get("type_") or "")
        retired = type_name.endswith(".LatticeOptVoxelNode") or type_name == "com.cad.sim.lattice_voxel"
        if not retired and not type_name.endswith(f".{TOPOLOGY_SOLVER_CLASS_NAME}"):
            continue
        custom = record.setdefault("custom", {})
        family = normalize_family_key(custom.get("structure_mode"))
        if family and family != "solid":
            custom.setdefault("post_lattice_type", custom["structure_mode"])
        for old, new in (
            ("lattice_cell_size_mm", "post_lattice_cell_size_mm"),
            ("lattice_member_thickness_mm", "post_lattice_min_thickness_mm"),
            ("lattice_skin_thickness_mm", "post_lattice_skin_thickness_mm"),
            ("lattice_min_relative_density", "post_lattice_min_density"),
            ("lattice_max_relative_density", "post_lattice_max_density"),
        ):
            if old in custom:
                custom.setdefault(new, custom[old])
        if retired:
            record["type_"] = f"{TOPOLOGY_SOLVER_IDENTIFIER}.{TOPOLOGY_SOLVER_CLASS_NAME}"
            record["name"] = str(record.get("name") or "Topology Solver").replace("Lattice Solver", "Topology Solver", 1)
            custom["penal"] = 3.0
        for name in {"structure_mode"} | _LATTICE_ONLY_PROPERTIES:
            custom.pop(name, None)
    return session


def _reject_legacy_independent_loads(session: dict[str, Any]) -> None:
    """Refuse to silently combine an explicitly saved independent case set."""
    nodes = session.get("nodes")
    if not isinstance(nodes, dict):
        return
    connections = session.get("connections") or []
    for node_id, record in nodes.items():
        if not _is_topology_node(record):
            continue
        custom = record.get("custom", {})
        if not isinstance(custom, dict):
            continue
        if custom.get("load_case_mode") != "Multiple Independent Cases":
            continue
        count = sum(
            isinstance(connection, dict) and connection.get("in") == [node_id, "loads"]
            for connection in connections
        )
        if count > 1:
            raise ValueError(
                "This saved topology study explicitly uses independent load cases. "
                "Split them into separate studies, or change the saved load_case_mode "
                "to Simultaneous Superposition to explicitly combine them."
            )


def compact_design_studio_session(session: dict[str, Any]) -> dict[str, Any]:
    """Preserve inspector settings, including advanced edits and face layers."""
    migrate_lattice_topology_nodes(session)
    _reject_legacy_independent_loads(session)
    compact = deepcopy(session)
    compact.pop("_guided_topology_policy", None)
    return compact


def migrate_legacy_box_properties(session: dict[str, Any]) -> dict[str, Any]:
    """Normalize interim BoxNode aliases to the registered x/y/z properties."""
    nodes = session.get("nodes")
    if not isinstance(nodes, dict):
        return session
    for record in nodes.values():
        if not isinstance(record, dict):
            continue
        type_name = str(record.get("type_") or "")
        if "BoxNode" in type_name or type_name.endswith(".box"):
            custom = record.get("custom")
            if isinstance(custom, dict):
                for alias, registered_name in (
                    ("length", "length_x"),
                    ("width", "width_y"),
                    ("height", "height_z"),
                ):
                    if alias in custom:
                        if registered_name not in custom:
                            custom[registered_name] = custom[alias]
                        custom.pop(alias, None)
    return session


_LEGACY_PORT_RENAMES = {
    "shape_out": "shape",
    "mesh_out": "mesh",
    "filepath_in": "file",
    "shape_in": "shape",
    "area_out": "area",
    "workplane": "selection",
    "assembly": "shape",
    "part_1": "bodies",
    "part_2": "bodies",
    "part_3": "bodies",
    "part_4": "bodies",
}


def migrate_legacy_ports_and_nodes(session: dict[str, Any]) -> dict[str, Any]:
    """Clean up retired ports and legacy properties from saved sessions.

    Migrates connections referencing retired ports (shape_out -> shape,
    workplane -> selection, part_1..part_4 -> bodies, etc.) and pops retired
    properties like fuse_parts on Group Bodies.
    """
    from pylcss.cae.design_studio.crash.conditions import migrate_impact_scenario_properties

    migrate_impact_scenario_properties(session)
    nodes = session.get("nodes")
    retired_topology_ports = {"preserved_regions", "obstacle_regions", "thermal_sinks", "thermal_loads"}
    retired_by_node = {}
    if isinstance(nodes, dict):
        for node_id, record in nodes.items():
            if not isinstance(record, dict):
                continue
            type_name = str(record.get("type_") or "")
            retired_ports = retired_topology_ports if _is_topology_node(record) else set()
            if type_name.endswith((".MaterialNode", ".material")):
                retired_ports = {"thermal_conductivity"}
            if retired_ports:
                retired_by_node[node_id] = retired_ports
                if isinstance(record.get("inputs"), dict):
                    for port_name in retired_ports:
                        record["inputs"].pop(port_name, None)
                if isinstance(record.get("input_ports"), list):
                    record["input_ports"] = [port for port in record["input_ports"]
                                             if port.get("name") not in retired_ports]
            custom = record.get("custom")
            if isinstance(custom, dict):
                if type_name.endswith((".MaterialNode", ".material")):
                    custom.pop("thermal_conductivity", None)
                if "AssemblyNode" in type_name or type_name.endswith(".assembly"):
                    custom.pop("fuse_parts", None)
                if type_name.endswith((".SolverNode", ".CrashSolverNode")):
                    from pylcss.input_values import as_bool
                    if "deck_only" not in custom and "run_external_solver" in custom:
                        custom["deck_only"] = not as_bool(custom["run_external_solver"])
                    custom.pop("run_external_solver", None)
                    custom.pop("solver_backend", None)
                if type_name.endswith(".CrashSolverNode"):
                    for name in (
                        "damping_alpha", "damping_beta", "enable_corotation",
                        "enable_contact", "contact_stiffness", "contact_thickness",
                        "contact_update_interval", "mass_scaling_threshold",
                        "impactor_mass_kg",
                    ):
                        custom.pop(name, None)

    connections = session.get("connections")
    if isinstance(connections, list):
        connections[:] = [connection for connection in connections if not (
            isinstance(connection, dict)
            and isinstance(connection.get("in"), list)
            and len(connection["in"]) == 2
            and connection["in"][1] in retired_by_node.get(connection["in"][0], set())
        )]
        for conn in connections:
            if not isinstance(conn, dict):
                continue
            in_info = conn.get("in")
            if isinstance(in_info, list) and len(in_info) == 2:
                if in_info[1] in _LEGACY_PORT_RENAMES:
                    in_info[1] = _LEGACY_PORT_RENAMES[in_info[1]]
            out_info = conn.get("out")
            if isinstance(out_info, list) and len(out_info) == 2:
                if out_info[1] in _LEGACY_PORT_RENAMES:
                    out_info[1] = _LEGACY_PORT_RENAMES[out_info[1]]

    return session


def expand_guided_topology_session(session: dict[str, Any]) -> dict[str, Any]:
    """Hydrate derived Guided controls before NodeGraphQt deserialization.

    This preserves the compact file format without making execution depend on
    whatever low-level defaults happen to be declared in the node constructor.
    A policy upgrade changes one deterministic mapping, while the saved design
    goal, fidelity, process, physical sizes, and user-visible choices remain
    authoritative.
    """
    migrate_lattice_topology_nodes(session)
    _reject_legacy_independent_loads(session)
    session = migrate_legacy_box_properties(session)
    session = migrate_legacy_ports_and_nodes(session)
    nodes = session.get("nodes")
    if not isinstance(nodes, dict):
        return session

    from pylcss.cae.design_studio.topology_optimization.configuration.presets import (
        industrial_topopt_defaults,
        migrate_design_goal,
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
        custom["physics_mode"] = "Structural"
        goal, aggregation = migrate_design_goal(custom.get("design_goal"))
        custom["design_goal"] = goal
        if aggregation is not None:
            custom["load_aggregation"] = aggregation
        legacy_workflow = custom.pop("workflow_mode", None)
        if str(legacy_workflow or "").lower() == "expert":
            custom.setdefault("automatic_grid", False)
            custom.setdefault("automatic_filter_radius", False)
            custom.setdefault("advanced_overrides", json.dumps(sorted(custom)))
            continue
        resolved = industrial_topopt_defaults(custom.get("design_goal"))
        # The former Guided policy always used these stopping controls. Keep
        # that behavior on migration while allowing new advanced edits to persist.
        if legacy_workflow is not None:
            custom.update(max_iter=120, tol=0.0035, convergence_patience=6)
        resolved.pop("workflow_mode", None)
        record["custom"] = {**resolved, **custom}
    return session


def sanitize_unregistered_nodes(
    session: dict[str, Any],
    known_classes: dict[str, type] | None = None,
) -> list[str]:
    """Gracefully omit nodes and dangling connections that have no registered class.

    If someone opens a .cad file that contains nodes that are no longer installed
    or registered (e.g. retired convenience nodes or uninstalled plugins), this
    strips those nodes and their connected wires cleanly from the session before
    deserialization, allowing the rest of the project to open smoothly.

    Returns:
        List of omitted node display names/labels for user notification.
    """
    nodes = session.get("nodes")
    if not isinstance(nodes, dict):
        return []

    if known_classes is None:
        try:
            from pylcss.cae.design_studio.node_library import NODE_CLASS_MAPPING

            classes = set(NODE_CLASS_MAPPING.values())
        except Exception:
            return []
    else:
        classes = set(known_classes.values())

    # Build the set of valid type strings
    valid_types = {"nodeGraphQt.nodes.BackdropNode", "BackdropNode"}
    for cls in classes:
        valid_types.add(f"{cls.__identifier__}.{cls.__name__}")
        valid_types.add(cls.__identifier__)
        valid_types.add(cls.__name__)

    omitted_ids: set[str] = set()
    omitted_descriptions: list[str] = []

    for node_id, node_data in list(nodes.items()):
        if not isinstance(node_data, dict):
            continue
        type_str = str(node_data.get("type_", "")).strip()
        if type_str not in valid_types:
            omitted_ids.add(str(node_id))
            node_name = str(node_data.get("name") or node_id)
            omitted_descriptions.append(f"{node_name} ({type_str})")
            del nodes[node_id]

    if omitted_ids:
        connections = session.get("connections")
        if isinstance(connections, list):
            cleaned_connections = []
            for conn in connections:
                if not isinstance(conn, dict):
                    continue
                out_id = str(conn.get("out", [""])[0])
                in_id = str(conn.get("in", [""])[0])
                if out_id not in omitted_ids and in_id not in omitted_ids:
                    cleaned_connections.append(conn)
            session["connections"] = cleaned_connections

    return omitted_descriptions
