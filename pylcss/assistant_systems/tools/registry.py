# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Agent Tools - Registry of available tools for agents.

Each tool has:
- Schema (for LLM function calling)
- Handler (actual execution)
- Validator (optional pre-execution check)
"""

from typing import TYPE_CHECKING
import logging

from pylcss.assistant_systems.tools.graph_validation import (
    run_cad_verified,
    run_system_verified,
    verify_cad_graph,
    verify_system_graph,
)
from pylcss.assistant_systems.tools.cad_node_catalog import CAD_NODE_TYPES
from pylcss.assistant_systems.tools.tool_types import (
    ParameterType,
    Tool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
)

if TYPE_CHECKING:
    from pylcss.assistant_systems.api.dispatcher import CommandDispatcher

__all__ = [
    "CAD_NODE_TYPES",
    "ParameterType",
    "Tool",
    "ToolCategory",
    "ToolParameter",
    "ToolRegistry",
    "create_pylcss_tools",
]

logger = logging.getLogger(__name__)


def create_pylcss_tools(command_dispatcher: "CommandDispatcher") -> ToolRegistry:
    """Create tool registry with all PyLCSS tools."""
    registry = ToolRegistry()

    # Graph payloads are normalized and verified before dispatch.

    # === CAD Tools ===

    registry.register(
        Tool(
            name="create_cad_geometry",
            description=(
                "Create or update geometry using the real Design Studio node library.\n\n"
                "CORE IDEA: Prefer GUI-native `com.cad.geometry.*` nodes for boxes, "
                "cylinders, tubes, holes, fillets, transforms, patterns, and booleans. "
                "For expert geometry that these nodes cannot express, use one or more "
                "`com.cad.code_part` nodes. Use the full expressive power of CadQuery — "
                "Workplane().box(), .hole(), .fillet(), .chamfer(), .extrude(), .revolve(), "
                ".cut(), .union(), .intersect(), faces() selectors, etc.\n\n"
                "CODE-PART RULE: Every tuneable dimension used by `com.cad.code_part` "
                "belongs in its `parameters` field, not hard-coded inside `code`. "
                "The `parameters` string is `name=value\\n...` lines; "
                "those names are then available as bare identifiers inside the code.\n\n"
                "EXAMPLE — bracket with a hole, one code_part:\n"
                '  nodes=[{"id":"bracket","type":"com.cad.code_part",'
                '"properties":{"code":"body=cq.Workplane(\'XY\').box(L,W,H)\\n'
                "body=body.faces('>Z').workplane().hole(hole_d)\\nresult=body\","
                '"parameters":"L=80\\nW=40\\nH=20\\nhole_d=10"}}]\n\n'
                "EXAMPLE — assembly of several parts, MULTIPLE code_part nodes "
                "(preferred when parts have independent parameters or get reused):\n"
                "  Create one `com.cad.code_part` per part. Connect their `shape` "
                "output ports into `com.cad.assembly` input ports. Each part stays "
                "independently parametric and the LLM can edit one part at a time.\n\n"
                "EXAMPLE — adding a hole to existing native geometry: insert a "
                "`com.cad.geometry.through_hole` node and connect shape → shape. "
                "A code_part is self-contained and does not accept an upstream shape.\n\n"
                "OTHER AVAILABLE TYPES (for FEA/IO only):\n"
                "  com.cad.assembly, com.cad.select_face,\n"
                "  com.cad.sim.material, com.cad.sim.mesh, com.cad.sim.constraint,\n"
                "  com.cad.sim.load, com.cad.sim.solver, com.cad.sim.topopt_voxel,\n"
                "  com.cad.topopt.support, com.cad.topopt.load, "
                "com.cad.topopt.non_design_region, "
                "com.cad.topopt.joint,\n"
                "  com.cad.topopt.operating_case, com.cad.topopt.thermal_sink, "
                "com.cad.topopt.heat_load,\n"
                "  com.cad.number, com.cad.variable, com.cad.export_step, com.cad.export_stl"
            ),
            parameters=[
                ToolParameter(
                    "nodes",
                    "array",
                    "List of nodes. Each node: {id, type, properties}. "
                    "Prefer GUI-native com.cad.geometry.* types for ordinary "
                    "engineering shapes. Use com.cad.code_part only when the "
                    "native nodes cannot express the requested geometry.",
                ),
                ToolParameter(
                    "connections",
                    "array",
                    "Connections list. Each: {from: 'node_id.output_port', to: 'node_id.input_port'}. "
                    "Wire shape outputs into assembly inputs, or into FEA mesh/constraint/load nodes.",
                    required=False,
                ),
            ],
            handler=lambda data: run_cad_verified(data, command_dispatcher),
            category="cad",
        )
    )

    registry.register(
        Tool(
            name="verify_cad_graph_json",
            description="Verify and sanitize CAD graph JSON without executing it. Use this before creating complex geometry.",
            parameters=[
                ToolParameter("nodes", "array", "List of CAD node specs to check"),
                ToolParameter(
                    "connections", "array", "CAD connections to check", required=False
                ),
                ToolParameter(
                    "goal",
                    "string",
                    "Original user goal for semantic verification",
                    required=False,
                ),
                ToolParameter(
                    "target_tool",
                    "string",
                    "Tool that will consume the verified payload",
                    required=False,
                ),
            ],
            handler=lambda data: verify_cad_graph(data),
            category="cad",
        )
    )

    registry.register(
        Tool(
            name="modify_cad_node",
            description=(
                "Modify properties of an existing Design Studio node by its exact "
                "name or stable ID. Invalid property names and enum values are "
                "reported instead of being silently ignored."
            ),
            parameters=[
                ToolParameter("node_id", "string", "The name/ID of the node to modify"),
                ToolParameter(
                    "properties", "object", 'Properties to update (e.g. {"width": 100})'
                ),
            ],
            handler=lambda data: command_dispatcher._build_node_graph(
                {
                    "params": {
                        "nodes": [
                            {
                                "id": data.get("node_id"),
                                "properties": data.get("properties", {}),
                            }
                        ]
                    }
                },
                sync=True,
            ),
            category="cad",
        )
    )

    registry.register(
        Tool(
            name="connect_cad_nodes",
            description="Connect two CAD nodes together. Use this to wire outputs to FEA node inputs (e.g. shape → sim.mesh, sim.mesh → sim.constraint, etc.).",
            parameters=[
                ToolParameter("from_node", "string", "Source node ID"),
                ToolParameter(
                    "from_port", "string", "Source port name (usually 'shape')"
                ),
                ToolParameter("to_node", "string", "Target node ID"),
                ToolParameter(
                    "to_port",
                    "string",
                    "Target port name (e.g. 'shape_a', 'shape_b', 'shape')",
                ),
            ],
            handler=lambda data: command_dispatcher._connect_nodes(
                {
                    "params": {
                        "from_node": data.get("from_node"),
                        "from_port": data.get("from_port"),
                        "to_node": data.get("to_node"),
                        "to_port": data.get("to_port"),
                    }
                },
                sync=True,
            ),
            category="cad",
        )
    )

    registry.register(
        Tool(
            name="execute_cad",
            description=(
                "Preview geometry or start one Design Studio workflow. If the graph "
                "contains several FEA, crash, or topology terminals, terminal_node "
                "is required so unrelated workflows are not run."
            ),
            parameters=[
                ToolParameter(
                    "terminal_node",
                    "string",
                    "Exact name or stable ID of the solver terminal to run.",
                    required=False,
                ),
                ToolParameter(
                    "preview",
                    "boolean",
                    "If true, update CAD/mesh previews without launching solvers.",
                    required=False,
                    default=False,
                ),
            ],
            handler=lambda data: command_dispatcher._cad_execute_scoped(
                data.get("terminal_node"),
                preview=bool(data.get("preview", False)),
                sync=True,
            ),
            category="cad",
        )
    )

    registry.register(
        Tool(
            name="stop_cad",
            description="Request a safe stop of the active Design Studio solver run.",
            parameters=[],
            handler=lambda data: command_dispatcher._cad_stop(sync=True),
            category="cad",
        )
    )

    registry.register(
        Tool(
            name="get_design_studio_state",
            description=(
                "Inspect Design Studio nodes, connections, independent solver "
                "workflows, cached-result availability, and current run state."
            ),
            parameters=[],
            handler=lambda data: command_dispatcher._get_cad_state(sync=True),
            category="cad",
        )
    )

    registry.register(
        Tool(
            name="select_design_studio_node",
            description=(
                "Select a Design Studio node by exact name or stable ID and show "
                "its cached geometry or simulation result in the 3D viewer."
            ),
            parameters=[
                ToolParameter(
                    "node",
                    "string",
                    "Exact node name or stable ID returned by get_design_studio_state.",
                ),
            ],
            handler=lambda data: command_dispatcher._cad_select_node(
                data.get("node", ""), sync=True
            ),
            category="cad",
        )
    )

    registry.register(
        Tool(
            name="save_design_studio_project",
            description=(
                "Save the complete Design Studio graph to an explicit .cad path, "
                "including cached FEA, crash, and topology results in its safe "
                "HDF5 sidecar."
            ),
            parameters=[
                ToolParameter(
                    "filename",
                    "string",
                    "Destination .cad filename or absolute path.",
                ),
            ],
            handler=lambda data: command_dispatcher._cad_save_project_file(
                data.get("filename", ""), sync=True
            ),
            category="cad",
            requires_confirmation=True,
        )
    )

    registry.register(
        Tool(
            name="load_design_studio_project",
            description=(
                "Load a Design Studio .cad graph and restore any saved numerical "
                "result sidecar without automatically running a solver."
            ),
            parameters=[
                ToolParameter(
                    "filename",
                    "string",
                    "Existing .cad filename or absolute path.",
                ),
            ],
            handler=lambda data: command_dispatcher._cad_load_project_file(
                data.get("filename", ""), sync=True
            ),
            category="cad",
            requires_confirmation=True,
        )
    )

    registry.register(
        Tool(
            name="create_freecad_part",
            description=(
                "Insert a `com.cad.freecad_part` node into the CAD graph for "
                "interactive (mouse-driven) authoring in FreeCAD.\n\n"
                "When to use this instead of `create_cad_geometry`:\n"
                "- The user explicitly asks to 'open in FreeCAD', 'sketch by hand', "
                "'draw it interactively', or wants to define loads / boundary "
                "conditions through FreeCAD's FEM workbench UI.\n"
                "- The geometry has a complex sketch (compound profile, splines, "
                "fillets following an irregular path) that's faster to draw than "
                "to write as CadQuery code.\n\n"
                "After insertion, tell the user to **double-click the node** to "
                "open the FreeCAD GUI. When they save inside FreeCAD, PyLCSS "
                "auto-imports the geometry + named selections + FEM definitions "
                "via the BREP + sidecar JSON the startup macro writes.\n\n"
                "Requires FreeCAD to be installed locally: "
                "`python scripts/install_solvers.py --only freecad`."
            ),
            parameters=[
                ToolParameter(
                    "name",
                    "string",
                    "Node display name shown in the graph (e.g. 'bracket', 'fork'). "
                    "Also seeds the .FCStd filename under data_freecad/.",
                ),
            ],
            handler=lambda data: command_dispatcher._build_node_graph(
                {
                    "params": {
                        "nodes": [
                            {
                                "id": data.get("name") or "freecad_part",
                                "type": "com.cad.freecad_part",
                                "properties": {},
                            }
                        ],
                    }
                },
                sync=True,
            ),
            category="cad",
        )
    )

    registry.register(
        Tool(
            name="export_cad",
            description=(
                "Export the selected or last-rendered cached CAD/recovered topology "
                "shape to an explicit STL or STEP path."
            ),
            parameters=[
                ToolParameter(
                    "format", "string", "Export format", enum=["stl", "step"]
                ),
                ToolParameter(
                    "filename", "string", "Destination filename or absolute path"
                ),
            ],
            handler=lambda data: command_dispatcher._cad_export_file(
                data.get("format", "step"),
                data.get("filename", ""),
                sync=True,
            ),
            category="cad",
            requires_confirmation=True,
        )
    )

    # === Modeling Tools ===

    registry.register(
        Tool(
            name="create_system_model",
            description=(
                "Create a system model with inputs, outputs, functions, and intermediates.\n"
                "Node types: com.pfd.input (Design Variable), com.pfd.output (QoI), "
                "com.pfd.intermediate (pass-through), com.pfd.custom_block (Python function).\n"
                "Port naming: Input/Output/Intermediate ports = var_name. CustomBlock ports = in_1, in_2, ... / out_1, out_2, ...\n"
                'Connection format: {"from": "node_id.port_name", "to": "node_id.port_name"}\n'
                'Example: {"from": "width.width", "to": "calc.in_1"} connects InputNode \'width\' to CustomBlock \'calc\''
            ),
            parameters=[
                ToolParameter(
                    "nodes",
                    "array",
                    "List of system nodes. Each: {id, type, properties}. "
                    "Input props: var_name, unit, min, max. "
                    "Output props: var_name, unit, req_min, req_max, minimize, maximize. "
                    "Intermediate props: var_name, unit. "
                    "CustomBlock props: num_inputs, num_outputs, code_content.",
                ),
                ToolParameter(
                    "connections",
                    "array",
                    "Connections: [{from: 'nodeId.portName', to: 'nodeId.portName'}]. "
                    "Use var_name as port for I/O/Intermediate. Use in_1/out_1 for CustomBlock.",
                    required=False,
                ),
            ],
            handler=lambda data: run_system_verified(data, command_dispatcher),
            category="modeling",
        )
    )

    registry.register(
        Tool(
            name="verify_system_graph_json",
            description="Verify system-model graph JSON without executing it. Use this before creating complex models.",
            parameters=[
                ToolParameter(
                    "nodes", "array", "List of system-model node specs to check"
                ),
                ToolParameter(
                    "connections",
                    "array",
                    "System-model connections to check",
                    required=False,
                ),
                ToolParameter(
                    "target_tool",
                    "string",
                    "Tool that will consume the verified payload",
                    required=False,
                ),
            ],
            handler=lambda data: verify_system_graph(data),
            category="modeling",
        )
    )

    registry.register(
        Tool(
            name="add_input_variable",
            description="Add a design variable (input) to the system model. Output port is named after var_name.",
            parameters=[
                ToolParameter(
                    "name", "string", "Variable name (becomes the output port name)"
                ),
                ToolParameter("min", "number", "Minimum value in the design space"),
                ToolParameter("max", "number", "Maximum value in the design space"),
                ToolParameter(
                    "unit",
                    "string",
                    "Physical unit (pint-compatible, e.g. 'mm', 'kg', 'N/m^2', '-' for dimensionless)",
                    required=False,
                    default="-",
                ),
            ],
            handler=lambda data: command_dispatcher._build_system_graph(
                {
                    "params": {
                        "nodes": [
                            {
                                "id": data.get("name", "input"),
                                "type": "com.pfd.input",
                                "properties": {
                                    "var_name": data.get("name"),
                                    "min": data.get("min", 0),
                                    "max": data.get("max", 100),
                                    "unit": data.get("unit", "-"),
                                },
                            }
                        ]
                    }
                },
                sync=True,
            ),
            category="modeling",
        )
    )

    registry.register(
        Tool(
            name="modify_system_node",
            description="Modify properties of a system node (names, min/max, code_content).",
            parameters=[
                ToolParameter("node_id", "string", "ID/Name of the node to modify"),
                ToolParameter(
                    "properties",
                    "object",
                    "Map of properties to update (e.g. {'var_name': 'x2', 'min': 5, 'code_content': '# code...'})",
                ),
            ],
            handler=lambda data: command_dispatcher._modify_system_node(
                {"params": data}, sync=True
            ),
            category="modeling",
        )
    )

    registry.register(
        Tool(
            name="get_graph_state",
            description="Get the current state of the graph (nodes, IDs, properties). Use this before editing to find node IDs.",
            parameters=[],
            handler=lambda data: command_dispatcher._get_graph_state(sync=True),
            category="modeling",
        )
    )

    registry.register(
        Tool(
            name="add_output_variable",
            description="Add an output (Quantity of Interest / QoI) to the system model. "
            "Set requirements with req_min/req_max. Set minimize=True or maximize=True for optimization objectives. "
            "Input port is named after var_name.",
            parameters=[
                ToolParameter(
                    "name",
                    "string",
                    "Output variable name (becomes the input port name)",
                ),
                ToolParameter(
                    "unit",
                    "string",
                    "Physical unit (pint-compatible, e.g. 'mm', 'kg', '-')",
                    required=False,
                    default="-",
                ),
                ToolParameter(
                    "req_min",
                    "number",
                    "Requirement lower bound (use -1e9 for unconstrained)",
                    required=False,
                ),
                ToolParameter(
                    "req_max",
                    "number",
                    "Requirement upper bound (use 1e9 for unconstrained)",
                    required=False,
                ),
                ToolParameter(
                    "minimize",
                    "boolean",
                    "Set True to minimize this output (optimization objective)",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    "maximize",
                    "boolean",
                    "Set True to maximize this output (optimization objective)",
                    required=False,
                    default=False,
                ),
            ],
            handler=lambda data: command_dispatcher._build_system_graph(
                {
                    "params": {
                        "nodes": [
                            {
                                "id": data.get("name", "output"),
                                "type": "com.pfd.output",
                                "properties": {
                                    k: v
                                    for k, v in {
                                        "var_name": data.get("name"),
                                        "unit": data.get("unit", "-"),
                                        "req_min": data.get("req_min"),
                                        "req_max": data.get("req_max"),
                                        "minimize": data.get("minimize"),
                                        "maximize": data.get("maximize"),
                                    }.items()
                                    if v is not None
                                },
                            }
                        ]
                    }
                },
                sync=True,
            ),
            category="modeling",
        )
    )

    registry.register(
        Tool(
            name="validate_model",
            description="Validate the current system model for errors and missing connections.",
            parameters=[],
            handler=lambda data: command_dispatcher._validate_graph(sync=True),
            category="modeling",
        )
    )

    registry.register(
        Tool(
            name="build_model",
            description="Build and transfer the system model for analysis (DOE, optimization, sensitivity).",
            parameters=[],
            handler=lambda data: command_dispatcher._build_model(),
            category="modeling",
        )
    )

    registry.register(
        Tool(
            name="clear_graph",
            description="Clear all nodes from the current graph.",
            parameters=[],
            handler=lambda data: command_dispatcher._clear_graph(sync=True),
            category="modeling",
            requires_confirmation=True,
        )
    )

    # === Analysis Tools ===

    registry.register(
        Tool(
            name="run_sensitivity_analysis",
            description="Run Sobol sensitivity analysis to identify which design variables most affect the outputs.",
            parameters=[],
            handler=lambda data: command_dispatcher._run_sensitivity(),
            category="analysis",
        )
    )

    registry.register(
        Tool(
            name="get_sensitivity_results",
            description="Retrieve results from the last sensitivity analysis showing variable importance rankings.",
            parameters=[],
            handler=lambda data: command_dispatcher._get_sensitivity_results(),
            category="analysis",
        )
    )

    registry.register(
        Tool(
            name="run_optimization",
            description="Run optimization to find optimal design parameters that minimize/maximize objectives.",
            parameters=[],
            handler=lambda data: command_dispatcher._run_optimization(),
            category="analysis",
        )
    )

    registry.register(
        Tool(
            name="stop_optimization",
            description="Stop a running optimization.",
            parameters=[],
            handler=lambda data: command_dispatcher._stop_optimization(),
            category="analysis",
        )
    )

    registry.register(
        Tool(
            name="train_surrogate",
            description="Train a surrogate model (ML approximation) for a specific function node.",
            parameters=[
                ToolParameter(
                    "node_name", "string", "Name of the node to train surrogate for"
                ),
            ],
            handler=lambda data: command_dispatcher._train_surrogate_node(
                {"params": data}
            ),
            category="analysis",
        )
    )

    registry.register(
        Tool(
            name="generate_samples",
            description="Generate samples in the solution space using Design of Experiments (DOE).",
            parameters=[],
            handler=lambda data: command_dispatcher._generate_samples(),
            category="analysis",
        )
    )

    # === Navigation Tools ===

    TAB_MAP = {
        "modeling": 0,
        "cad": 1,
        "surrogate": 2,
        "solution_space": 3,
        "optimization": 4,
        "sensitivity": 5,
    }

    registry.register(
        Tool(
            name="switch_tab",
            description="Switch to a different application tab/environment.",
            parameters=[
                ToolParameter(
                    "tab", "string", "Tab to switch to", enum=list(TAB_MAP.keys())
                ),
            ],
            handler=lambda data: command_dispatcher._handle_switch_tab(
                {"tab": TAB_MAP.get(data.get("tab", "cad"), 1)}
            ),
            category="navigation",
        )
    )

    # === Project Tools ===

    registry.register(
        Tool(
            name="save_project",
            description="Save the current project.",
            parameters=[],
            handler=lambda data: command_dispatcher._save_project(),
            category="project",
        )
    )

    registry.register(
        Tool(
            name="new_project",
            description="Create a new project (clears current work).",
            parameters=[],
            handler=lambda data: command_dispatcher._new_project(),
            category="project",
            requires_confirmation=True,
        )
    )

    logger.info(f"Created tool registry with {len(registry.all_tools)} tools")
    return registry
