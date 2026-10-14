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
    is_lattice_study_record,
)

# Retired lattice settings translated or removed before node construction.
_LATTICE_ONLY_PROPERTIES = frozenset(
    {
        "lattice_cell_size_mm",
        "lattice_member_thickness_mm",
        "lattice_skin_thickness_mm",
        "lattice_target_relative_density",
        "lattice_porosity",
        "repair_lattice_connections",
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
        "validate_after_optimize",
        "generate_cad_after_optimize",
        "validation_quality",
        "draw_direction",
        "extrusion_axis",
        "manufacturing_process",
        "workflow_mode",
        "yield_stress",
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


_RETIRED_INFILL_PROPERTIES = frozenset({
    "automatic_grid", "voxel_mesh_quality", "voxel_size", "max_voxel_elements",
    "automatic_mesh_refinement", "automatic_filter_radius", "advanced_overrides",
    "design_goal", "physics_mode", "formulation", "advanced_settings_visible",
    "validation_displacement_limit_mm", "validation_yield_safety_factor",
    "topology_convergence_enabled", "topology_convergence_levels", "volfrac", "rmin",
    "penal", "E0", "Emin", "nu", "optimizer", "load_aggregation", "load_pnorm_p",
    "max_iter", "tol", "convergence_patience", "stress_constraint",
    "structure_cell_size_voxels", "structure_member_thickness_voxels",
    "structure_skin_thickness_voxels", "lattice_settings_mode", "lattice_variable_density",
    "lattice_cell_size_mm", "lattice_member_thickness_mm", "lattice_skin_thickness_mm",
    "lattice_min_relative_density", "lattice_max_relative_density",
    "lattice_solid_transition_density", "lattice_porosity",
    "post_lattice_type", "post_lattice_cell_size_mm", "post_lattice_min_thickness_mm",
    "post_lattice_skin_thickness_mm", "post_lattice_min_density", "post_lattice_max_density",
})


def _migrate_infill_properties(custom):
    # Older zero skin meant Automatic. New None/Automatic/Manual choices are explicit.
    if "infill_skin_mode" not in custom:
        try:
            thickness = float(custom.get("infill_skin_thickness_mm") or 0.0)
        except (TypeError, ValueError):
            thickness = -1.0
        custom["infill_skin_mode"] = "Automatic" if thickness == 0 else "Manual"
    custom.setdefault("infill_allow_pitch_coarsening", True)
    if custom.get("infill_cell_size_mode") == "Manual" and not custom.get("infill_cell_size_mm"):
        custom["infill_cell_size_mode"] = "Automatic"
    for name in _RETIRED_INFILL_PROPERTIES | _RETIRED_TOPOLOGY_PROPERTIES:
        custom.pop(name, None)
    for name in list(custom):
        if name.startswith("fea_"):
            custom.pop(name)


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


def migrate_cad_reconstruction_properties(session: dict[str, Any]) -> dict[str, Any]:
    """Normalize retired reconstruction selections before graph deserialization."""
    # Structure capability lookup does not load the CAD kernel.
    nodes = session.get("nodes")
    if isinstance(nodes, dict):
        for record in nodes.values():
            if not isinstance(record, dict):
                continue
            custom = record.get("custom")
            if isinstance(custom, dict):
                custom.pop("cad_reconstruction_method", None)
                density_node = any(
                    name in str(record.get("type_", ""))
                    for name in ("topopt_voxel", "lattice_voxel", "lattice_infill")
                )
                if (
                    density_node
                    and "visualization" in custom
                    and custom["visualization"]
                    not in {"CAD", "STL Shape", "Lattice", "Density"}
                    and not str(custom["visualization"]).startswith("FEA ")
                ):
                    custom["visualization"] = "STL Shape"
                if not density_node:
                    continue
                from .topology_optimization.geometry.lattice_cad import (
                    topopt_export_formats,
                )

                formats = topopt_export_formats(custom.get("structure_mode"))
                if (
                    custom.get("cad_export_format")
                    and custom["cad_export_format"].upper() not in formats
                ):
                    custom["cad_export_format"] = formats[0]
                    filename = custom.get("cad_export_filename")
                    if filename:
                        from pathlib import Path

                        custom["cad_export_filename"] = str(
                            Path(filename).with_suffix(
                                "." + custom["cad_export_format"].lower()
                            )
                        )
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
        if not is_density_study_record(record):
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
    compact = deepcopy(session)
    migrate_lattice_topology_nodes(compact)
    _reject_legacy_independent_loads(compact)
    migrate_fem_properties(compact)
    compact.pop("_guided_topology_policy", None)
    return compact


def migrate_authoring_properties(session: dict[str, Any]) -> dict[str, Any]:
    """Convert retired authoring properties once, before node construction."""
    nodes = session.get("nodes")
    if not isinstance(nodes, dict):
        return session
    for record in nodes.values():
        if not isinstance(record, dict):
            continue
        type_name = str(record.get("type_") or "")
        custom = record.get("custom")
        if not isinstance(custom, dict):
            continue
        if type_name.endswith((".NumberNode", ".VariableNode")) or type_name in {
            "NumberNode", "VariableNode", "com.cad.number", "com.cad.variable",
        }:
            if "value_input" not in custom and "value" in custom:
                custom["value_input"] = str(custom["value"])
            custom.pop("value", None)
        if type_name.endswith(".InteractiveSelectFaceNode") or type_name in {
            "InteractiveSelectFaceNode", "com.cad.select_face_interactive",
        }:
            if "picked_face_indices" in custom:
                custom.setdefault("picked_entity_indices", custom.pop("picked_face_indices"))
        if "BoxNode" in type_name or type_name.endswith(".box"):
            for alias, registered_name in (
                ("length", "length_x"),
                ("width", "width_y"),
                ("height", "height_z"),
            ):
                if alias in custom:
                    custom.setdefault(registered_name, custom.pop(alias))
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


def migrate_impact_scenario_properties(session: dict[str, Any]) -> bool:
    """Normalize retired impact nodes, ports, and physical settings on import."""
    nodes = session.get("nodes", {}) if isinstance(session, dict) else {}
    if not isinstance(nodes, dict):
        return False
    before = json.dumps(session, sort_keys=True)
    connections = session.get("connections", [])
    if not isinstance(connections, list):
        connections = []

    def kind(record):
        return str(record.get("type_", "")) if isinstance(record, dict) else ""

    def rename_ports(record, direction, names):
        ports = record.get(direction)
        if isinstance(ports, dict):
            for old, new in names.items():
                if old not in ports:
                    continue
                value = ports.pop(old)
                existing = ports.setdefault(new, value)
                if isinstance(existing, dict) and isinstance(value, dict):
                    for node_id, connected in value.items():
                        current = existing.setdefault(node_id, [])
                        for port in connected:
                            if port not in current:
                                current.append(port)
        declarations = record.get(direction[:-1] + "_ports")
        if isinstance(declarations, list):
            seen = set()
            normalized = []
            for port in declarations:
                if not isinstance(port, dict):
                    normalized.append(port)
                    continue
                port["name"] = names.get(port.get("name"), port.get("name"))
                if port["name"] not in seen:
                    seen.add(port["name"])
                    normalized.append(port)
            record[direction[:-1] + "_ports"] = normalized

    for record in nodes.values():
        type_name = kind(record)
        if not isinstance(record, dict):
            continue
        custom = record.setdefault("custom", {})
        if not isinstance(custom, dict):
            continue
        if type_name.endswith(
            (".CrashMaterialNode", ".ImpactMaterialNode")
        ) or type_name.startswith("com.cad.sim.crash_material"):
            record["type_"] = "com.cad.sim.material.MaterialNode"
            custom["analysis_purpose"] = "Impact"
            type_name = record["type_"]
        if type_name.endswith(".MaterialNode"):
            rename_ports(
                record,
                "outputs",
                {"impact_material": "material", "crash_material": "material"},
            )
        if type_name.endswith(".CrashSolverNode"):
            rename_ports(
                record,
                "inputs",
                {"impact_material": "material", "crash_material": "material"},
            )
            rename_ports(record, "outputs", {"crash_results": "results"})
        if type_name.endswith(".ImpactConditionNode"):
            scope = str(custom.get("application_scope", "")).lower().replace("_", " ")
            if scope == "impact face":
                custom["application_scope"] = "Fixed specimen + moving impactor"
            elif scope == "moving body":
                custom["application_scope"] = "Moving body + fixed wall"
            try:
                if float(custom.get("wall_friction", 0.0)) < 0.0:
                    custom["wall_friction"] = (
                        0.0 if scope.startswith("moving body") else 0.08
                    )
            except (TypeError, ValueError):
                pass

    # NodeGraphQt also stores connection names inside node port dictionaries.
    for record in nodes.values():
        if not isinstance(record, dict):
            continue
        for direction in ("inputs", "outputs"):
            ports = record.get(direction, {})
            if not isinstance(ports, dict):
                continue
            for linked in ports.values():
                if not isinstance(linked, dict):
                    continue
                for node_id, port_names in linked.items():
                    remote = kind(nodes.get(str(node_id)))
                    names = {}
                    if direction == "outputs" and remote.endswith(".CrashSolverNode"):
                        names = {
                            "impact_material": "material",
                            "crash_material": "material",
                        }
                    elif direction == "inputs":
                        if remote.endswith(".MaterialNode"):
                            names = {
                                "impact_material": "material",
                                "crash_material": "material",
                            }
                        elif remote.endswith(".CrashSolverNode"):
                            names = {"crash_results": "results"}
                    if isinstance(port_names, list):
                        linked[node_id] = list(
                            dict.fromkeys(names.get(name, name) for name in port_names)
                        )

    normalized_connections = []
    seen = set()
    for connection in connections:
        if not isinstance(connection, dict):
            normalized_connections.append(connection)
            continue
        output_ref, input_ref = connection.get("out"), connection.get("in")
        if (
            not isinstance(output_ref, (list, tuple))
            or not isinstance(input_ref, (list, tuple))
            or len(output_ref) != 2
            or len(input_ref) != 2
        ):
            normalized_connections.append(connection)
            continue
        source = nodes.get(str(output_ref[0]))
        target = nodes.get(str(input_ref[0]))
        output_ref, input_ref = list(output_ref), list(input_ref)
        if kind(source).endswith(".MaterialNode") and output_ref[1] in {
            "impact_material",
            "crash_material",
        }:
            output_ref[1] = "material"
            source["custom"]["analysis_purpose"] = "Impact"
        if (
            kind(source).endswith(".CrashSolverNode")
            and output_ref[1] == "crash_results"
        ):
            output_ref[1] = "results"
        if kind(target).endswith(".CrashSolverNode") and input_ref[1] in {
            "impact_material",
            "crash_material",
        }:
            input_ref[1] = "material"
        if (
            kind(source).endswith(".ImpactConditionNode")
            and kind(target).endswith(".CrashSolverNode")
            and output_ref[1] == input_ref[1] == "impact"
        ):
            scenario_custom, solver_custom = source["custom"], target["custom"]
            if solver_custom.get("impactor_mass_kg") is not None:
                scenario_custom.setdefault(
                    "impactor_mass_kg", solver_custom["impactor_mass_kg"]
                )
            if "enable_contact" in solver_custom:
                scenario_custom.setdefault(
                    "self_contact_enabled", solver_custom["enable_contact"]
                )
                scenario_custom.setdefault("self_contact_friction", 0.08)
        connection.update(out=output_ref, **{"in": input_ref})
        identity = tuple(output_ref + input_ref)
        if identity not in seen:
            seen.add(identity)
            normalized_connections.append(connection)
    if "connections" in session:
        session["connections"] = normalized_connections
    for record in nodes.values():
        if kind(record).endswith(".CrashSolverNode") and isinstance(
            record.get("custom"), dict
        ):
            record["custom"].pop("impactor_mass_kg", None)
            record["custom"].pop("enable_contact", None)
    return before != json.dumps(session, sort_keys=True)


def migrate_fem_properties(session: object) -> bool:
    """Normalize retired FEM settings before graph deserialization."""
    if not isinstance(session, dict) or not isinstance(session.get("nodes"), dict):
        return False
    support_names = {
        "Pinned": "Fixed",
        "Pinned (Fixed for solids)": "Fixed",
        **{
            f"{kind} {axis}": f"Block {axis} Translation"
            for kind in ("Roller", "Symmetry") for axis in "XYZ"
        },
    }
    changed = False
    for record in session["nodes"].values():
        if not isinstance(record, dict) or not isinstance(record.get("custom"), dict):
            continue
        type_name = str(record.get("type_") or "")
        custom = record["custom"]
        retired = ()
        if type_name.endswith(".MeshNode") or type_name in {
            "MeshNode", "com.cad.sim.mesh",
        }:
            retired = ("mesher", "gmsh_algorithm", "curvature_elements")
        elif type_name.endswith(".LoadNode") or type_name in {
            "LoadNode", "com.cad.sim.load",
        }:
            retired = ("moment_x", "moment_y", "moment_z")
        elif type_name.endswith(".ConstraintNode") or type_name in {
            "ConstraintNode", "com.cad.sim.constraint",
        }:
            support = custom.get("constraint_type")
            if isinstance(support, str) and support in support_names:
                custom["constraint_type"] = support_names[support]
                changed = True
        for name in retired:
            if name in custom:
                custom.pop(name)
                changed = True
    return changed


def migrate_legacy_ports_and_nodes(session: dict[str, Any]) -> dict[str, Any]:
    """Clean up retired ports and legacy properties from saved sessions.

    Migrates connections referencing retired ports (shape_out -> shape,
    workplane -> selection, part_1..part_4 -> bodies, etc.) and pops retired
    properties like fuse_parts on Group Bodies.
    """
    migrate_authoring_properties(session)
    migrate_impact_scenario_properties(session)
    migrate_fem_properties(session)
    nodes = session.get("nodes")
    retired_topology_ports = {"preserved_regions", "obstacle_regions", "thermal_sinks", "thermal_loads"}
    retired_by_node = {}
    if isinstance(nodes, dict):
        for node_id, record in nodes.items():
            if not isinstance(record, dict):
                continue
            type_name = str(record.get("type_") or "")
            retired_ports = retired_topology_ports if is_density_study_record(record) else set()
            if is_lattice_study_record(record):
                retired_ports = retired_ports | {"material", "supports", "loads"}
            if is_density_study_record(record):
                outputs = record.get("outputs")
                if isinstance(outputs, dict):
                    for retired_name in ("result", "topopt_result"):
                        if retired_name in outputs:
                            outputs.setdefault("results", outputs.pop(retired_name))
                for port in record.get("output_ports") or []:
                    if isinstance(port, dict) and port.get("name") in {"result", "topopt_result"}:
                        port["name"] = "results"
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
                if out_info[1] in {"result", "topopt_result"} and isinstance(nodes, dict) and is_density_study_record(nodes.get(out_info[0])):
                    out_info[1] = "results"
                if out_info[1] in _LEGACY_PORT_RENAMES:
                    out_info[1] = _LEGACY_PORT_RENAMES[out_info[1]]

    return session


def expand_guided_topology_session(session: dict[str, Any]) -> dict[str, Any]:
    """Fill missing solver defaults before NodeGraphQt deserialization.

    Saved user settings override defaults derived from the current design goal.
    """
    migrate_lattice_topology_nodes(session)
    _reject_legacy_independent_loads(session)
    session = migrate_legacy_ports_and_nodes(session)
    nodes = session.get("nodes")
    if not isinstance(nodes, dict):
        return session

    from pylcss.cae.design_studio.topology_optimization.configuration.presets import (
        industrial_topopt_defaults,
        normalize_design_goal,
    )

    for record in nodes.values():
        if not is_density_study_record(record):
            continue
        custom = record.get("custom")
        if not isinstance(custom, dict):
            continue
        if is_lattice_study_record(record):
            _migrate_infill_properties(custom)
            continue
        # Remove retired/legacy property keys that are no longer registered
        # NodeGraphQt properties. An unknown name is not ignored on load — it
        # raises and takes the whole project with it — so a property that is
        # withdrawn has to be listed here in the same change that withdraws it.
        for retired in _RETIRED_TOPOLOGY_PROPERTIES:
            custom.pop(retired, None)
        custom["physics_mode"] = "Structural"
        custom["design_goal"] = normalize_design_goal(custom.get("design_goal"))
        resolved = industrial_topopt_defaults(custom["design_goal"])
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


def prepare_design_studio_session(
    session: dict[str, Any],
    known_classes: dict[str, type] | None = None,
) -> list[str]:
    """Apply the common GUI/runtime import policy and return omitted node labels.

    Reconstruction settings are normalized before retired lattice nodes are
    retyped; expansion then migrates ports and fills missing solver defaults.
    The supplied session is updated in place, ready for deserialization.
    """
    migrate_cad_reconstruction_properties(session)
    expand_guided_topology_session(session)
    return sanitize_unregistered_nodes(session, known_classes=known_classes)
