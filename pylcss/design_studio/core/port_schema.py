# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Semantic port contracts for the Design Studio graph.

NodeGraphQt identifies ports by their serialized names.  Those names must stay
stable so existing ``.cad`` projects keep loading. This module adds UI labels
and data types without depending on Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from pylcss.design_studio.topology_optimization.integration.study_identity import (
    LATTICE_INFILL_CLASS_NAME,
    LATTICE_SOLVER_CLASS_NAME,
    TOPOLOGY_SOLVER_CLASS_NAME,
    is_density_study_class,
)


@dataclass(frozen=True)
class PortDescriptor:
    """Human-facing contract for one node port."""

    label: str
    data_type: str
    required: bool
    description: str


_TYPE_DESCRIPTIONS = {
    "Geometry": "CAD body or surface geometry",
    "Solid Geometry": "Closed CAD solid or watertight volume",
    "Surface Geometry": "CAD surface or shell midsurface",
    "Design Space": "Optimizable solid geometry",
    "Selection": "Named face, edge, vertex, or mesh-entity selection",
    "Region Scope": "Named selection or explicit CAD region geometry",
    "Material": "Structural material definition",
    "Impact Material": "Explicit-impact material definition",
    "Mesh": "Finite-element mesh",
    "Solid Mesh": "Tetrahedral solid finite-element mesh",
    "Shell Mesh": "Triangular shell finite-element mesh",
    "Body": "Mesh paired with a material",
    "Support": "Structural support or prescribed displacement",
    "Structural Load": "Force, pressure, or gravity load",
    "Topology Support": "Topology-optimization support definition",
    "Topology Load": "Topology-optimization load definition",
    "Topology Region": "Preserved solid or void region",
    "Joint": "Kinematic joint definition",
    "Load Case": "Grouped topology operating case",
    "Thermal Boundary": "Thermal sink or heat-load definition",
    "Impact Setup": "Initial motion and contact",
    "Result Dataset": "Solved fields, histories, quality checks, and summaries",
    "Engineering Result": "Structural, impact, or topology result data",
    "Topology Result": "Topology density and recovered geometry",
    "Scalar": "Numeric value without a declared physical dimension",
    "Dimensionless": "Dimensionless numeric value",
    "Integer": "Whole-number count",
    "Length": "Length in model units",
    "Area": "Area in model units squared",
    "Volume": "Volume in model units cubed",
    "Force": "Force",
    "Pressure": "Pressure, stress, or elastic modulus",
    "Density": "Mass density",
    "Acceleration": "Acceleration",
    "Mass": "Mass",
    "Energy / Mass": "Specific energy",
    "Percent": "Percentage",
    "Thermal Conductivity": "Thermal conductivity",
    "Table": "Structured engineering data",
    "File": "File reference",
    "Any": "Unconstrained data",
}

_PORT_LABELS = {
    "shape": "Geometry",
    "shape_a": "Geometry A",
    "shape_b": "Geometry B",
    "shape_in": "Geometry",
    "shape_out": "Geometry",
    "base": "Base Geometry",
    "tool": "Tool Geometry",
    "bodies": "Bodies",
    "design_domain": "Design Domain",
    "region_shape": "Region Volume",
    "recovered_shape": "Recovered Surface",
    "selection": "Selection",
    "workplane": "Selection (Legacy)",
    "target_face": "Target Selection",
    "target_region": "Target Selection",
    "refinement_faces": "Local Refinement Selection",
    "anchor_a": "Anchor Selection A",
    "anchor_b": "Anchor Selection B",
    "impact_face": "Impact Surface",
    "mesh": "Mesh",
    "material": "Structural Material",
    "impact_material": "Impact Material",
    "component": "Body",
    "components": "Bodies",
    "constraints": "Supports",
    "supports": "Supports",
    "loads": "Loads",
    "impact": "Impact Setup",
    "results": "Results",
    "result_view": "View",
    "topology_results": "Result",
    "topopt_result": "Topology Result (Legacy)",
    "value": "Value",
    "result": "Result",
    "properties": "Mass Properties",
    "dimensions": "Dimensions",
    "file": "File",
}

_EXPLICIT_LABELS = {
    ("TopologySupportNode", "input", "target_region"): "Support Interface (Face)",
    ("TopologySupportNode", "output", "supports"): "Support",
    ("TopologyLoadNode", "input", "target_region"): "Load Interface (Face)",
    ("TopologyLoadNode", "output", "loads"): "Force",
    ("TopologyOptVoxelNode", "input", "design_domain"): "Design Domain",
    ("TopologyOptVoxelNode", "input", "material"): "Material",
    ("TopologyOptVoxelNode", "input", "supports"): "Supports",
    ("TopologyOptVoxelNode", "input", "loads"): "Forces",
    ("TopologyOptVoxelNode", "output", "results"): "Optimization Result",
    ("TopologyOptVoxelNode", "output", "recovered_shape"): "Recovered Surface",
    ("ConstraintNode", "input", "mesh"): "FE Mesh (FEA)",
    ("ConstraintNode", "input", "target_face"): "Selection",
    ("ConstraintNode", "output", "constraints"): "Support",
    ("LoadNode", "input", "mesh"): "FE Mesh (FEA)",
    ("LoadNode", "input", "target_face"): "Selection",
    ("LoadNode", "output", "loads"): "Load",
    ("PressureLoadNode", "input", "mesh"): "FE Mesh (FEA)",
    ("PressureLoadNode", "input", "target_face"): "Selection",
    ("PressureLoadNode", "output", "loads"): "Load",
    ("SolverNode", "output", "results"): "Result",
    ("CrashSolverNode", "output", "results"): "Result",
    ("NumberNode", "output", "value"): "Value",
    ("VariableNode", "output", "value"): "Value",
    ("MathExpressionNode", "output", "result"): "Value",
    ("BoxNode", "input", "length"): "Length (mm)",
    ("BoxNode", "input", "width"): "Width (mm)",
    ("BoxNode", "input", "height"): "Height (mm)",
    ("CylinderNode", "input", "diameter"): "Diameter (mm)",
    ("CylinderNode", "input", "length"): "Length (mm)",
    ("TubeNode", "input", "outer_diameter"): "Outer Diameter (mm)",
    ("TubeNode", "input", "wall_thickness"): "Wall (mm)",
    ("TubeNode", "input", "length"): "Length (mm)",
    ("CylindricalShellNode", "input", "diameter"): "Diameter (mm)",
    ("CylindricalShellNode", "input", "length"): "Length (mm)",
    ("ThroughHoleNode", "input", "diameter"): "Diameter (mm)",
    ("FilletNode", "input", "radius"): "Radius (mm)",
    ("MassPropertiesNode", "output", "mass"): "Mass (kg)",
    ("MassPropertiesNode", "output", "volume"): "Volume (mm³)",
    ("BoundingBoxNode", "output", "length"): "Length (mm)",
    ("BoundingBoxNode", "output", "width"): "Width (mm)",
    ("BoundingBoxNode", "output", "height"): "Height (mm)",
    ("BoundingBoxNode", "output", "volume"): "Box Volume (mm³)",
    ("MeasureDistanceNode", "output", "distance"): "Distance (mm)",
    ("SurfaceAreaNode", "output", "area"): "Area (mm²)",
    ("MaterialNode", "input", "youngs_modulus"): "Young's Modulus (MPa)",
    ("MaterialNode", "input", "poissons_ratio"): "Poisson's Ratio",
    ("MaterialNode", "input", "density"): "Density (t/mm³)",
    (
        "MaterialNode",
        "input",
        "thermal_conductivity",
    ): "Thermal Conductivity (mW/mm/K)",
    ("LoadNode", "input", "force_x"): "Force X (N)",
    ("LoadNode", "input", "force_y"): "Force Y (N)",
    ("LoadNode", "input", "force_z"): "Force Z (N)",
    ("PressureLoadNode", "input", "pressure"): "Pressure (MPa)",
    ("MeshNode", "input", "element_size"): "Element Size (mm)",
    ("MeshNode", "input", "refinement_size"): "Local Size (mm)",
}

_NUMERIC_NAME_TYPES = {
    "accel": "Acceleration",
    "area": "Area",
    "count": "Integer",
    "density": "Density",
    "diameter": "Length",
    "distance": "Length",
    "element_size": "Length",
    "force_x": "Force",
    "force_y": "Force",
    "force_z": "Force",
    "height": "Length",
    "length": "Length",
    "mass": "Mass",
    "outer_diameter": "Length",
    "poissons_ratio": "Dimensionless",
    "pressure": "Pressure",
    "radius": "Length",
    "refinement_size": "Length",
    "spacing": "Length",
    "thermal_conductivity": "Thermal Conductivity",
    "thickness": "Length",
    "volume": "Volume",
    "wall_thickness": "Length",
    "width": "Length",
    "youngs_modulus": "Pressure",
}

_GENERIC_SCALAR_PORT_NAMES = {
    "param_1",
    "param_2",
    "param_3",
    "param_4",
    "param_5",
    "param_6",
    "value",
    "x",
    "y",
    "z",
}

_GEOMETRY_PORT_NAMES = {
    "base",
    "bodies",
    "design_domain",
    "part_1",
    "part_2",
    "part_3",
    "part_4",
    "recovered_shape",
    "region_shape",
    "shape",
    "shape_a",
    "shape_b",
    "shape_in",
    "shape_out",
    "tool",
}

_SELECTION_PORT_NAMES = {
    "anchor_a",
    "anchor_b",
    "impact_face",
    "refinement_faces",
    "selection",
    "target_face",
    "target_region",
}

_EXPLICIT_TYPES = {
    ("TopologyOptVoxelNode", "input", "design_domain"): "Design Space",
    # Legacy selection output is deliberately not geometry.
    ("SelectFaceNode", "output", "workplane"): "Selection",
    ("InteractiveSelectFaceNode", "output", "workplane"): "Selection",
    # ``loads`` means different things in structural and topology workflows.
    ("LoadNode", "output", "loads"): "Structural Load",
    ("PressureLoadNode", "output", "loads"): "Structural Load",
    ("SolverNode", "input", "loads"): "Structural Load",
    ("ConstraintNode", "output", "constraints"): "Support",
    ("SolverNode", "input", "constraints"): "Support",
    ("CrashSolverNode", "input", "constraints"): "Support",
    ("TopologySupportNode", "output", "supports"): "Topology Support",
    ("TopologyOptVoxelNode", "input", "supports"): "Topology Support",
    ("TopologyLoadNode", "output", "loads"): "Topology Load",
    ("TopologyOptVoxelNode", "input", "loads"): "Topology Load",
    ("ImpactConditionNode", "output", "impact"): "Impact Setup",
    ("CrashSolverNode", "input", "impact"): "Impact Setup",
    ("MaterialNode", "output", "impact_material"): "Impact Material",
    ("MaterialNode", "output", "crash_material"): "Impact Material",
    ("CrashSolverNode", "input", "material"): "Material",
    ("CrashSolverNode", "input", "impact_material"): "Impact Material",
    ("TopologyOptVoxelNode", "output", "results"): "Topology Result",
    ("TopologyOptVoxelNode", "output", "topopt_result"): "Topology Result",
    ("BoxNode", "output", "shape"): "Solid Geometry",
    ("CylinderNode", "output", "shape"): "Solid Geometry",
    ("TubeNode", "output", "shape"): "Solid Geometry",
    ("CylindricalShellNode", "output", "shape"): "Surface Geometry",
    ("BooleanNode", "input", "base"): "Solid Geometry",
    ("BooleanNode", "input", "tool"): "Solid Geometry",
    ("BooleanNode", "output", "shape"): "Solid Geometry",
    ("ThroughHoleNode", "input", "shape"): "Solid Geometry",
    ("ThroughHoleNode", "output", "shape"): "Solid Geometry",
    ("FilletNode", "input", "shape"): "Solid Geometry",
    ("FilletNode", "output", "shape"): "Solid Geometry",
    ("SolverNode", "input", "mesh"): "Solid Mesh",
    ("MassPropertiesNode", "output", "mass"): "Mass",
    ("MassPropertiesNode", "output", "volume"): "Volume",
    ("BoundingBoxNode", "output", "length"): "Length",
    ("BoundingBoxNode", "output", "width"): "Length",
    ("BoundingBoxNode", "output", "height"): "Length",
    ("BoundingBoxNode", "output", "volume"): "Volume",
}

_NAME_TYPES = {
    "material": "Material",
    "mesh": "Mesh",
    "component": "Body",
    "components": "Body",
    "bodies": "Geometry",
    "constraints": "Support",
    "impact": "Impact Setup",
    "impact_material": "Impact Material",
    "results": "Result Dataset",
    "result_view": "Result Dataset",
    "topology_results": "Topology Result",
    "topopt_result": "Topology Result",
    "file": "File",
    "filepath_in": "File",
    "properties": "Table",
    "dimensions": "Table",
    "result": "Scalar",
}

_ACCEPTED_OUTPUT_TYPES = {
    "Geometry": {
        "Geometry",
        "Solid Geometry",
        "Surface Geometry",
        "Design Space",
    },
    "Solid Geometry": {"Solid Geometry", "Geometry", "Design Space"},
    "Surface Geometry": {"Surface Geometry", "Geometry"},
    "Region Scope": {
        "Selection",
        "Geometry",
        "Solid Geometry",
        "Surface Geometry",
    },
    # Direct geometry connections remain valid for projects created before the
    # explicit Design Space node was introduced.
    "Design Space": {"Design Space", "Geometry", "Solid Geometry"},
    "Engineering Result": {"Result Dataset", "Topology Result"},
    "Mesh": {"Mesh", "Solid Mesh", "Shell Mesh"},
    "Solid Mesh": {"Solid Mesh", "Mesh"},
    "Shell Mesh": {"Shell Mesh", "Mesh"},
    # Supports and surface loads are study definitions, not solver-specific
    # data.  The shared public Support/Force/Pressure nodes can feed either a
    # static FEA solve or the voxel optimizer; legacy Topology nodes remain
    # valid in saved projects.
    "Support": {"Support", "Topology Support"},
    "Topology Support": {"Topology Support", "Support"},
    "Structural Load": {"Structural Load", "Topology Load"},
    "Topology Load": {"Topology Load", "Structural Load"},
}

_NUMERIC_TYPES = {
    "Scalar",
    "Dimensionless",
    "Integer",
    "Length",
    "Area",
    "Volume",
    "Force",
    "Pressure",
    "Density",
    "Acceleration",
    "Mass",
    "Energy / Mass",
    "Percent",
    "Thermal Conductivity",
}

_REQUIRED_INPUTS = {
    "AssemblyNode": {"bodies"},
    "BooleanNode": {"base", "tool"},
    "BoundingBoxNode": {"shape"},
    "CrashSolverNode": {"mesh", "material", "impact"},
    "FEAComponentNode": {"mesh", "material"},
    "FilletNode": {"shape"},
    "ImpactConditionNode": {"impact_face"},
    "InteractiveSelectFaceNode": {"shape"},
    "LinearPatternNode": {"shape"},
    "MassPropertiesNode": {"shape"},
    "MeasureDistanceNode": {"shape_a", "shape_b"},
    "MeshNode": {"shape"},
    "PressureLoadNode": {"target_face"},
    "SelectFaceNode": {"shape"},
    "SurfaceAreaNode": {"shape"},
    "ThroughHoleNode": {"shape"},
    "TopologyLoadNode": {"target_region"},
    "TopologyOptVoxelNode": {"design_domain", "material"},
    "TopologySupportNode": {"target_region"},
    "TransformNode": {"shape"},
}


def _mirror_topology_study_entries(table: dict) -> None:
    """Give the other density studies the topology study's port vocabulary.

    ``LatticeOptVoxelNode`` and ``LatticeInfillNode`` both descend from the
    topology node and declare no ports of their own, so restating these tables
    by hand would only create a second place for the same label, type, and
    requirement to drift from. The infill deletes the ports it does not use
    rather than relabelling them, so the entries it inherits for those never
    resolve against a real port.
    """
    for key, value in list(table.items()):
        if isinstance(key, tuple) and key[0] == TOPOLOGY_SOLVER_CLASS_NAME:
            table[(LATTICE_SOLVER_CLASS_NAME, *key[1:])] = value
            table[(LATTICE_INFILL_CLASS_NAME, *key[1:])] = value
        elif key == TOPOLOGY_SOLVER_CLASS_NAME:
            table[LATTICE_SOLVER_CLASS_NAME] = value
            table[LATTICE_INFILL_CLASS_NAME] = value


for _study_table in (_EXPLICIT_LABELS, _EXPLICIT_TYPES, _REQUIRED_INPUTS):
    _mirror_topology_study_entries(_study_table)
del _study_table


def _node_class_name(node: object) -> str:
    return node.__class__.__name__


def _property(node: object, name: str, default: Any = None) -> Any:
    getter = getattr(node, "get_property", None)
    if not callable(getter):
        return default
    try:
        value = getter(name)
    except Exception:
        return default
    return default if value is None else value


def _has_connection(node: object, name: str) -> bool:
    getter = getattr(node, "get_input", None)
    if not callable(getter):
        return False
    try:
        port = getter(name)
        return bool(port and port.connected_ports())
    except Exception:
        return False


def human_port_label(name: str) -> str:
    """Return a stable engineering label for a serialized port name."""
    if name in _PORT_LABELS:
        return _PORT_LABELS[name]
    part_match = re.fullmatch(r"part_(\d+)", name)
    if part_match:
        return f"Body {part_match.group(1)}"
    return " ".join(
        token.upper() if token in {"x", "y", "z"} else token.title()
        for token in str(name).split("_")
    )


def engineering_port_label(node: object, name: str, direction: str) -> str:
    """Return a node-specific label while keeping serialized names stable."""
    param_match = re.fullmatch(r"param_(\d+)", str(name))
    if param_match:
        property_name = f"param_{param_match.group(1)}_name"
        configured = str(_property(node, property_name, "") or "").strip()
        if configured:
            return configured
    return _EXPLICIT_LABELS.get(
        (_node_class_name(node), direction, name),
        human_port_label(name),
    )


def semantic_port_type(node: object, name: str, direction: str) -> str:
    """Infer a semantic data type from node class, direction, and stable name."""
    class_name = _node_class_name(node)
    explicit = _EXPLICIT_TYPES.get((class_name, direction, name))
    if explicit:
        return explicit
    if class_name == "MeshNode":
        mesh_type = str(_property(node, "mesh_type", "Tet") or "Tet")
        if name == "shape" and direction == "input":
            return "Surface Geometry" if mesh_type == "Shell" else "Solid Geometry"
        if name == "mesh" and direction == "output":
            return "Shell Mesh" if mesh_type == "Shell" else "Solid Mesh"
    if name in _GEOMETRY_PORT_NAMES:
        return "Geometry"
    if name in _SELECTION_PORT_NAMES:
        return "Selection"
    if name in _NUMERIC_NAME_TYPES:
        return _NUMERIC_NAME_TYPES[name]
    if name in _GENERIC_SCALAR_PORT_NAMES or name.startswith("param_"):
        return "Scalar"
    return _NAME_TYPES.get(name, "Any")


def _port_label_item(node: object, port_view: object) -> Any:
    """Return the graphics item NodeGraphQt pairs with a port, if any.

    NodeGraphQt keeps each port's socket and its text label as two separate
    graphics items, indexed on the node view as ``{port_item: text_item}``.
    """
    view = getattr(node, "view", None)
    for attribute in ("_input_items", "_output_items"):
        items = getattr(view, attribute, None)
        if isinstance(items, dict) and port_view in items:
            return items[port_view]
    return None


def _set_port_visible(node: object, name: str, visible: bool) -> None:
    """Set one port's presentation state without changing serialization."""
    for getter_name in ("get_input", "get_output"):
        getter = getattr(node, getter_name, None)
        if not callable(getter):
            continue
        try:
            port = getter(name)
        except Exception:
            port = None
        if port is None:
            continue
        try:
            is_visible = bool(visible)
            setter = getattr(port, "set_visible", None)
            current = getattr(port, "visible", None)
            if (
                callable(setter)
                and callable(current)
                and bool(current()) != is_visible
            ):
                setter(is_visible, push_undo=False)
            else:
                port.view.setVisible(is_visible)
            # NodeGraphQt's zoom/proxy-mode switch restores every label whose
            # socket has ``display_name=True``, even if that socket itself is
            # hidden. Keep this flag aligned with contextual visibility so a
            # zoom cannot resurrect orphan text below the node.
            port.view.display_name = is_visible
            # The label is a sibling item that NodeGraphQt only repositions
            # while its port is visible. Hiding the socket alone therefore
            # stranded the text in the node body — a Structural topology study
            # still read "Heat Sinks" and "Heat Loads" under its last real
            # port, with nothing to connect to.
            label = _port_label_item(node, port.view)
            if label is not None:
                label.setVisible(is_visible)
        except Exception:
            pass
        return


def apply_context_port_visibility(node: object) -> None:
    """Hide ports that cannot affect the node's current operating mode."""
    class_name = _node_class_name(node)
    if class_name == "LoadNode":
        is_force = str(_property(node, "load_type", "Force") or "Force") != "Gravity"
        _set_port_visible(node, "target_face", is_force)
        for name in ("force_x", "force_y", "force_z"):
            _set_port_visible(node, name, is_force)
    elif class_name == "ImpactConditionNode":
        scenario = str(
            _property(node, "application_scope", "Fixed specimen + moving impactor")
            or ""
        )
        is_moving_body = scenario.strip().lower().replace("_", " ").startswith(
            "moving body"
        )
        _set_port_visible(node, "impact_face", not is_moving_body)
    elif is_density_study_class(class_name):
        physics = str(_property(node, "physics_mode", "Structural") or "").lower()
        structural = "structural" in physics or "coupled" in physics
        thermal = "thermal" in physics
        for name in ("supports", "loads"):
            _set_port_visible(node, name, structural)
    elif class_name == "AssemblyNode":
        _set_port_visible(node, "bodies", True)
        for index in range(1, 5):
            _set_port_visible(node, f"part_{index}", False)
    elif class_name == "FreeCadPartNode":
        # These ports are useful before the first FreeCAD save: a user can
        # wire Parameters while authoring, then the discovered Spreadsheet
        # aliases rename the sockets in place.  Hiding blank slots made them
        # impossible to reach and they were never reliably shown again after
        # aliases arrived from the background execution worker.
        for index in range(1, 9):
            _set_port_visible(node, f"param_{index}", True)


def is_required_input(node: object, name: str) -> bool:
    """Return whether an input is required for the node's current mode."""
    class_name = _node_class_name(node)
    if class_name == "ImpactConditionNode" and name == "impact_face":
        scenario = str(
            _property(node, "application_scope", "Fixed specimen + moving impactor")
        )
        return not scenario.strip().lower().replace("_", " ").startswith(
            "moving body"
        )
    if class_name == "LoadNode" and name == "target_face":
        return (
            str(_property(node, "load_type", "Force")) != "Gravity"
            and not str(_property(node, "condition", "")).strip()
        )
    if class_name == "ConstraintNode" and name == "target_face":
        return not str(_property(node, "condition", "")).strip()
    if class_name == "SolverNode":
        if name in {"mesh", "material"}:
            return not _has_connection(node, "components")
        return name in {"constraints", "loads"}
    if class_name == "AssemblyNode" and name == "bodies":
        legacy_connected = any(
            _has_connection(node, f"part_{index}") for index in range(1, 5)
        )
        return not legacy_connected
    if class_name == "CrashSolverNode" and name == "constraints":
        try:
            impact_port = node.get_input("impact")
            connected = list(impact_port.connected_ports()) if impact_port else []
            setup = connected[0].node() if connected else None
            scenario = str(
                _property(
                    setup,
                    "application_scope",
                    "Fixed specimen + moving impactor",
                )
                or ""
            )
        except Exception:
            scenario = "Fixed specimen + moving impactor"
        return scenario.strip().lower().replace("_", " ").startswith(
            "fixed specimen"
        )
    if is_density_study_class(class_name):
        if name in {"design_domain", "material"}:
            return True
        physics = str(_property(node, "physics_mode", "Structural")).lower()
        if name in {"supports", "loads"}:
            return "structural" in physics or "coupled" in physics
        return False
    return name in _REQUIRED_INPUTS.get(class_name, set())


def apply_display_port_labels(node: object) -> None:
    """Apply engineering labels without changing serialized port names."""
    apply_context_port_visibility(node)
    for getter_name in ("input_ports", "output_ports"):
        direction = "input" if getter_name == "input_ports" else "output"
        getter = getattr(node, getter_name, None)
        if not callable(getter):
            continue
        try:
            ports = getter()
            if isinstance(ports, dict):
                ports = ports.values()
        except Exception:
            continue
        for port in ports:
            try:
                raw_name = str(port.name())
                label = engineering_port_label(node, raw_name, direction)
                # ``PortItem.name`` is not presentation-only: NodeGraphQt
                # uses it as the key into the node's input/output mappings
                # when an interactive drag is released. Replacing it with
                # the engineering label made every labelled socket impossible
                # to connect (for example, ``shape`` became ``Geometry`` and
                # the graph lookup raised ``KeyError('Geometry')``). Keep the
                # view identity synchronized with the serialized model name
                # and update only its sibling text item.
                port.view.name = raw_name
                label_item = _port_label_item(node, port.view)
                if label_item is not None:
                    label_item.setPlainText(label)
            except Exception:
                continue
    try:
        node.view.draw_node()
    except Exception:
        pass


def describe_port(node: object, port: object, direction: str) -> PortDescriptor:
    """Build the inspector descriptor for a NodeGraphQt port."""
    raw_name = getattr(port, "name", lambda: "")()
    data_type = semantic_port_type(node, str(raw_name), direction)
    return PortDescriptor(
        label=engineering_port_label(node, str(raw_name), direction),
        data_type=data_type,
        required=direction == "input" and is_required_input(node, str(raw_name)),
        description=_TYPE_DESCRIPTIONS.get(data_type, "Graph data"),
    )


def validate_connection(input_port: object, output_port: object) -> tuple[bool, str]:
    """Validate a graph connection using semantic types.

    Unknown/legacy ports remain permissive.  This keeps old expert nodes usable
    while preventing known mistakes such as connecting a Material to Geometry.
    """
    try:
        input_node = input_port.node()
        output_node = output_port.node()
        input_name = str(input_port.name())
        output_name = str(output_port.name())
    except Exception:
        return True, ""

    input_type = semantic_port_type(input_node, input_name, "input")
    output_type = semantic_port_type(output_node, output_name, "output")
    if input_type in _NUMERIC_TYPES and output_type in _NUMERIC_TYPES:
        # Generic Scalar is the compatibility type used by Parameter and Math
        # Expression. Declared physical quantities must otherwise match.
        if "Scalar" in {input_type, output_type} or input_type == output_type:
            return True, ""
        input_label = engineering_port_label(input_node, input_name, "input")
        output_label = engineering_port_label(output_node, output_name, "output")
        return (
            False,
            f"Cannot connect {output_label} ({output_type}) to "
            f"{input_label} ({input_type}).",
        )
    accepted_output_types = _ACCEPTED_OUTPUT_TYPES.get(
        input_type,
        {input_type},
    )
    if "Any" in {input_type, output_type} or output_type in accepted_output_types:
        return True, ""

    input_label = engineering_port_label(input_node, input_name, "input")
    output_label = engineering_port_label(output_node, output_name, "output")
    return (
        False,
        f"Cannot connect {output_label} ({output_type}) to "
        f"{input_label} ({input_type}).",
    )


__all__ = [
    "PortDescriptor",
    "apply_context_port_visibility",
    "apply_display_port_labels",
    "describe_port",
    "engineering_port_label",
    "human_port_label",
    "is_required_input",
    "semantic_port_type",
    "validate_connection",
]
