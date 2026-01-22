# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Agent Tools - Registry of available tools for agents.

Each tool has:
- Schema (for LLM function calling)
- Handler (actual execution)
- Validator (optional pre-execution check)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import logging
from .tools.gear_tools import create_helical_gear

if TYPE_CHECKING:
    from .command_dispatcher import CommandDispatcher

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class Tool:
    """A tool that agents can invoke."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None
    category: str = "general"  # "cad", "modeling", "analysis", "navigation"
    requires_confirmation: bool = False
    validator: Optional[Callable[[Dict], bool]] = None
    
    def to_openai_schema(self) -> Dict:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
                
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
        
    def get_description_for_prompt(self) -> str:
        """Get a description suitable for inclusion in prompts."""
        params_desc = ", ".join([
            f"{p.name}: {p.type}" + (f" (optional)" if not p.required else "")
            for p in self.parameters
        ])
        return f"- **{self.name}**({params_desc}): {self.description}"


class ToolRegistry:
    """Registry of all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        
    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
        
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
        
    def list_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a category."""
        return [t for t in self._tools.values() if t.category == category]
        
    def get_all_schemas(self) -> List[Dict]:
        """Get OpenAI schemas for all tools."""
        return [t.to_openai_schema() for t in self._tools.values()]
        
    def get_category_schemas(self, category: str) -> List[Dict]:
        """Get schemas for tools in a category."""
        return [t.to_openai_schema() for t in self.list_by_category(category)]
        
    def get_tools_description(self, categories: Optional[List[str]] = None) -> str:
        """Get a text description of tools for prompts."""
        tools = self._tools.values()
        if categories:
            tools = [t for t in tools if t.category in categories]
            
        by_category: Dict[str, List[Tool]] = {}
        for tool in tools:
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)
            
        lines = []
        for category, category_tools in sorted(by_category.items()):
            lines.append(f"\n### {category.upper()} Tools")
            for tool in category_tools:
                lines.append(tool.get_description_for_prompt())
                
        return "\n".join(lines)
        
    @property
    def all_tools(self) -> List[Tool]:
        return list(self._tools.values())


def create_pylcss_tools(command_dispatcher: 'CommandDispatcher') -> ToolRegistry:
    """Create tool registry with all PyLCSS tools."""
    registry = ToolRegistry()
    
    # === CAD Tools ===
    
    registry.register(Tool(
        name="create_cad_geometry",
        description="Create CAD geometry using a node graph specification. \n\nExamples:\n- Basic: `nodes=[{'type': 'com.cad.box', 'id':'b1'}]`\n- Hollowing/Shelling: Create the main shape AND a smaller inner shape. Then use `connect_cad_nodes` with `com.cad.boolean` (operation='cut').\n- Drilling: Create a base shape AND a cylinder. Then use `com.cad.boolean` (operation='cut').",
        parameters=[
            ToolParameter("nodes", "array", "List of nodes. Each node has: id (string), type (e.g. 'com.cad.box'), properties (object with dimensions)"),
            ToolParameter("connections", "array", "List of connections. Each has: from ('node_id.port'), to ('node_id.port')", required=False),
        ],
        handler=lambda data: command_dispatcher._build_node_graph({"params": data}, sync=True),
        category="cad",
    ))
    
    registry.register(Tool(
        name="modify_cad_node",
        description="Modify properties of an existing CAD node by its name/ID.",
        parameters=[
            ToolParameter("node_id", "string", "The name/ID of the node to modify"),
            ToolParameter("properties", "object", "Properties to update (e.g. {\"width\": 100})"),
        ],
        handler=lambda data: command_dispatcher._build_node_graph({
            "params": {
                "nodes": [{"id": data.get("node_id"), "properties": data.get("properties", {})}]
            }
        }, sync=True),
        category="cad",
    ))
    
    registry.register(Tool(
        name="connect_cad_nodes",
        description="Connect two CAD nodes together. \n\nUsage:\n- Boolean: Connect shapes to `shape_a` and `shape_b` of `com.cad.boolean`.\n- Ops: Connect shape to `shape` input of modifier (fillet, etc.).",
        parameters=[
            ToolParameter("from_node", "string", "Source node ID"),
            ToolParameter("from_port", "string", "Source port name (usually 'shape')"),
            ToolParameter("to_node", "string", "Target node ID"),
            ToolParameter("to_port", "string", "Target port name (e.g. 'shape_a', 'shape_b', 'shape')"),
        ],
        handler=lambda data: command_dispatcher._connect_nodes({
            "params": {
                "from_node": data.get("from_node"),
                "from_port": data.get("from_port"),
                "to_node": data.get("to_node"),
                "to_port": data.get("to_port"),
            }
        }, sync=True),
        category="cad",
    ))
    
    registry.register(Tool(
        name="execute_cad",
        description="Execute the CAD graph to generate 3D geometry. Call this after creating or modifying nodes.",
        parameters=[],
        handler=lambda data: command_dispatcher._cad_execute(sync=True),
        category="cad",
    ))
    
    registry.register(Tool(
        name="export_cad",
        description="Export CAD geometry to STL or STEP file format.",
        parameters=[
            ToolParameter("format", "string", "Export format", enum=["stl", "step"]),
            ToolParameter("filename", "string", "Output filename (optional)", required=False),
        ],
        handler=lambda data: command_dispatcher._cad_export(sync=True),
        category="cad",
    ))

    registry.register(Tool(
        name="create_helical_gear",
        description="Create a helical gear (or spur gear). Generates a full CAD graph with correct tooth profile. Use this when asked for a 'gear' or 'gearbox'.",
        parameters=[
            ToolParameter("module", "number", "Gear module (size scaler). Default 1.0", required=False, default=1.0),
            ToolParameter("teeth", "number", "Number of teeth. Default 20", required=False, default=20),
            ToolParameter("width", "number", "Gear width. Default 10.0", required=False, default=10.0),
            ToolParameter("helix_angle", "number", "Helix angle in degrees. 0 for spur gear. Default 20.0", required=False, default=20.0),
            ToolParameter("pressure_angle", "number", "Pressure angle. Default 20.0", required=False, default=20.0),
            ToolParameter("center_x", "number", "X Position center. Default 0.0", required=False, default=0.0),
            ToolParameter("center_y", "number", "Y Position center. Default 0.0", required=False, default=0.0),
        ],
        handler=lambda data: command_dispatcher._build_node_graph(create_helical_gear(**data), sync=True),
        category="cad",
    ))
    
    # === Modeling Tools ===
    
    registry.register(Tool(
        name="create_system_model",
        description="Create a system model with inputs, outputs, functions, and intermediates.",
        parameters=[
            ToolParameter("nodes", "array", "List of system nodes. Types: com.pfd.input, com.pfd.output, com.pfd.intermediate, com.pfd.custom_block"),
            ToolParameter("connections", "array", "List of connections between nodes", required=False),
        ],
        handler=lambda data: command_dispatcher._build_system_graph({"params": data}, sync=True),
        category="modeling",
    ))
    
    registry.register(Tool(
        name="add_input_variable",
        description="Add a design variable (input) to the system model.",
        parameters=[
            ToolParameter("name", "string", "Variable name"),
            ToolParameter("min", "number", "Minimum value"),
            ToolParameter("max", "number", "Maximum value"),
            ToolParameter("default", "number", "Default value (optional)", required=False),
        ],
        handler=lambda data: command_dispatcher._build_system_graph({
            "params": {
                "nodes": [{
                    "id": data.get("name", "input"),
                    "type": "com.pfd.input",
                    "properties": {
                        "var_name": data.get("name"),
                        "min": data.get("min", 0),
                        "max": data.get("max", 100),
                    }
                }]
            }
        }, sync=True),
        category="modeling",
    ))
    
    registry.register(Tool(
        name="add_output_variable",
        description="Add an output (quantity of interest) to the system model.",
        parameters=[
            ToolParameter("name", "string", "Output variable name"),
        ],
        handler=lambda data: command_dispatcher._build_system_graph({
            "params": {
                "nodes": [{
                    "id": data.get("name", "output"),
                    "type": "com.pfd.output",
                    "properties": {"var_name": data.get("name")}
                }]
            }
        }, sync=True),
        category="modeling",
    ))
    
    registry.register(Tool(
        name="validate_model",
        description="Validate the current system model for errors and missing connections.",
        parameters=[],
        handler=lambda data: command_dispatcher._validate_graph(sync=True),
        category="modeling",
    ))
    
    registry.register(Tool(
        name="build_model",
        description="Build and transfer the system model for analysis (DOE, optimization, sensitivity).",
        parameters=[],
        handler=lambda data: command_dispatcher._build_model(),
        category="modeling",
    ))
    
    registry.register(Tool(
        name="clear_graph",
        description="Clear all nodes from the current graph.",
        parameters=[],
        handler=lambda data: command_dispatcher._clear_graph(sync=True),
        category="modeling",
    ))
    
    # === Analysis Tools ===
    
    registry.register(Tool(
        name="run_sensitivity_analysis",
        description="Run Sobol sensitivity analysis to identify which design variables most affect the outputs.",
        parameters=[],
        handler=lambda data: command_dispatcher._run_sensitivity(),
        category="analysis",
    ))
    
    registry.register(Tool(
        name="get_sensitivity_results",
        description="Retrieve results from the last sensitivity analysis showing variable importance rankings.",
        parameters=[],
        handler=lambda data: command_dispatcher._get_sensitivity_results(),
        category="analysis",
    ))
    
    registry.register(Tool(
        name="run_optimization",
        description="Run optimization to find optimal design parameters that minimize/maximize objectives.",
        parameters=[],
        handler=lambda data: command_dispatcher._run_optimization(),
        category="analysis",
    ))
    
    registry.register(Tool(
        name="stop_optimization",
        description="Stop a running optimization.",
        parameters=[],
        handler=lambda data: command_dispatcher._stop_optimization(),
        category="analysis",
    ))
    
    registry.register(Tool(
        name="train_surrogate",
        description="Train a surrogate model (ML approximation) for a specific function node.",
        parameters=[
            ToolParameter("node_name", "string", "Name of the node to train surrogate for"),
        ],
        handler=lambda data: command_dispatcher._train_surrogate_node({"params": data}),
        category="analysis",
    ))
    
    registry.register(Tool(
        name="generate_samples",
        description="Generate samples in the solution space using Design of Experiments (DOE).",
        parameters=[],
        handler=lambda data: command_dispatcher._generate_samples(),
        category="analysis",
    ))
    
    # === Navigation Tools ===
    
    TAB_MAP = {
        "modeling": 0, "cad": 1, "surrogate": 2,
        "solution_space": 3, "optimization": 4, "sensitivity": 5
    }
    
    registry.register(Tool(
        name="switch_tab",
        description="Switch to a different application tab/environment.",
        parameters=[
            ToolParameter("tab", "string", "Tab to switch to", enum=list(TAB_MAP.keys())),
        ],
        handler=lambda data: command_dispatcher._handle_switch_tab({
            "tab": TAB_MAP.get(data.get("tab", "cad"), 1)
        }),
        category="navigation",
    ))
    
    # === Project Tools ===
    
    registry.register(Tool(
        name="save_project",
        description="Save the current project.",
        parameters=[],
        handler=lambda data: command_dispatcher.mouse.hotkey('ctrl', 's'),
        category="project",
    ))
    
    registry.register(Tool(
        name="new_project",
        description="Create a new project (clears current work).",
        parameters=[],
        handler=lambda data: command_dispatcher._new_project(),
        category="project",
        requires_confirmation=True,
    ))
    
    logger.info(f"Created tool registry with {len(registry.all_tools)} tools")
    return registry


# === CAD Node Types Reference ===
# For use in prompts and documentation

CAD_NODE_TYPES = {
    # Primitives
    "com.cad.box": {"properties": ["box_length", "box_width", "box_depth"]},
    "com.cad.cylinder": {"properties": ["cyl_radius", "cyl_height"]},
    "com.cad.sphere": {"properties": ["radius"]},
    "com.cad.cone": {"properties": ["cone_radius1", "cone_radius2", "cone_height"]},
    "com.cad.torus": {"properties": ["torus_major_radius", "torus_minor_radius"]},
    "com.cad.wedge": {"properties": ["wedge_length", "wedge_width", "wedge_height"]},
    
    # Operations
    "com.cad.boolean": {"properties": ["operation"], "inputs": ["shape_a", "shape_b"]},
    "com.cad.fillet": {"properties": ["fillet_radius"], "inputs": ["shape"]},
    "com.cad.chamfer": {"properties": ["chamfer_distance"], "inputs": ["shape"]},
    "com.cad.shell": {"properties": ["shell_thickness"], "inputs": ["shape"]},
    
    # Transforms
    "com.cad.translate": {"properties": ["x_translate", "y_translate", "z_translate"], "inputs": ["shape"]},
    "com.cad.rotate": {"properties": ["axis", "angle"], "inputs": ["shape"]},
    "com.cad.mirror": {"properties": ["mirror_plane"], "inputs": ["shape"]},
    "com.cad.scale": {"properties": ["scale_factor"], "inputs": ["shape"]},
    
    # Patterns
    "com.cad.linear_pattern": {"properties": ["count", "spacing", "direction"], "inputs": ["shape"]},
    "com.cad.circular_pattern": {"properties": ["count", "angle", "axis"], "inputs": ["shape"]},
    
    # Advanced
    "com.cad.extrude": {"properties": ["extrude_height"], "inputs": ["sketch"]},
    "com.cad.revolve": {"properties": ["revolve_angle", "axis"], "inputs": ["sketch"]},
    "com.cad.loft": {"properties": [], "inputs": ["profiles"]},
    "com.cad.sweep": {"properties": [], "inputs": ["profile", "path"]},
}

SYSTEM_NODE_TYPES = {
    "com.pfd.input": {"properties": ["var_name", "min", "max", "default"]},
    "com.pfd.output": {"properties": ["var_name"]},
    "com.pfd.intermediate": {"properties": ["var_name"]},
    "com.pfd.custom_block": {"properties": ["code_content", "num_inputs", "num_outputs"]},
}


def get_cad_schema_for_prompt() -> str:
    """Get CAD node schema as text for LLM prompts."""
    lines = ["## Available CAD Node Types\n"]
    
    for node_type, info in CAD_NODE_TYPES.items():
        props = info.get("properties", [])
        inputs = info.get("inputs", ["shape"])
        lines.append(f"- **{node_type}**")
        if props:
            lines.append(f"  - Properties: {', '.join(props)}")
        lines.append(f"  - Inputs: {', '.join(inputs)}")
        lines.append(f"  - Outputs: shape")
        
    return "\n".join(lines)
