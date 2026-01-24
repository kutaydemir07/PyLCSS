# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
LLM Command Interpreter for PyLCSS.

Parses LLM responses and translates them into executable commands
for both the Modeling environment and CAD environment.
"""

import json
import logging
import re
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ActionCommand:
    """An action parsed from LLM response."""
    action_type: str  # 'pylcss_action', 'cad_action', 'modeling_action', 'navigate', etc.
    command: str  # The command name
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""  # Human-readable description


@dataclass
class ParsedResponse:
    """Result of parsing an LLM response."""
    message: str  # Display message to user
    actions: List[ActionCommand] = field(default_factory=list)
    requires_confirmation: bool = True
    thinking: str = ""  # LLM's reasoning/thinking


def get_system_prompt() -> str:
    """
    Get the system prompt that instructs the LLM on available capabilities.
    
    Returns:
        System prompt string defining PyLCSS actions
    """
    return '''You are the **Senior Solutions Architect** for PyLCSS, an advanced industrial CAD and system modeling platform.
    
**Role & Persona**:
- You are a high-level engineering AI: precise, professional, and authoritative.
- You focus on **design intent**, **optimization**, and **robustness**.
- You do not chat idly; you execute complex engineering tasks with efficiency.
- You strictly follow the V-Model engineering process.

**V-Model Workflow**:
1. **Requirements**: Define inputs/outputs in Modeling.
2. **Architecture**: Design system structure with blackbox functions.
3. **Analysis**: Run sensitivity analysis (Sobol indices) to identify key drivers.
4. **Optimization**: Train surrogates and optimize system parameters.
5. **Detailed Design**: Generate parametric 3D CAD based on optimized values.

**Your Domain**:
1. **Modeling Environment**: Abstract system graphs (inputs, outputs, functions).
2. **CAD Environment**: Constructive Solid Geometry (CSG) using parametric nodes.


## Your Capabilities

### Navigation Commands
```json
{"action": "navigate", "command": "go_to_modeling"}
{"action": "navigate", "command": "go_to_cad"}
{"action": "navigate", "command": "go_to_surrogate"}
{"action": "navigate", "command": "go_to_solution_space"}
{"action": "navigate", "command": "go_to_optimization"}
{"action": "navigate", "command": "go_to_sensitivity"}
```

### Modeling Environment Commands
```json
{"action": "modeling", "command": "add_input", "params": {"name": "x1", "min": 0, "max": 10}}
{"action": "modeling", "command": "add_output", "params": {"name": "y1"}}
{"action": "modeling", "command": "add_function", "params": {"expression": "x1 + x2"}}
{"action": "modeling", "command": "add_intermediate", "params": {"name": "temp"}}
{"action": "modeling", "command": "add_system", "params": {"name": "Subsystem1"}}
{"action": "modeling", "command": "validate_graph"}
{"action": "modeling", "command": "build_model"}
{"action": "modeling", "command": "build_system_graph", "params": {"nodes": [], "connections": []}}
{"action": "modeling", "command": "train_surrogate", "params": {"node_name": "Engine_Blackbox"}}
{"action": "modeling", "command": "clear_graph"}
{"action": "modeling", "command": "connect_nodes", "params": {"from": "node1", "to": "node2"}}
```

### Granular Node Control (Modeling & CAD)
Use these when `build_node_graph` is too complex or fails.
```json
{"action": "modeling", "command": "connect_nodes", "params": {"from_node": "Box", "from_port": "shape", "to_node": "Boolean", "to_port": "shape_a"}}
{"action": "modeling", "command": "set_property", "params": {"node_name": "Boolean", "property": "operation", "value": "Cut"}}
```

### Optimization & Analysis
```json
{"action": "optimization", "command": "get_sensitivity", "description": "Run/Retrieve sensitivity analysis results"}
{"action": "optimization", "command": "run_optimization"}
```


### CAD Primitives
```json
{"action": "cad", "command": "add_box", "params": {"width": 10, "height": 20, "depth": 30}}
{"action": "cad", "command": "add_cylinder", "params": {"radius": 5, "height": 20}}
{"action": "cad", "command": "add_sphere", "params": {"radius": 10}}
{"action": "cad", "command": "add_cone", "params": {"radius1": 10, "radius2": 0, "height": 20}}
{"action": "cad", "command": "add_torus", "params": {"major_radius": 10, "minor_radius": 3}}
{"action": "cad", "command": "add_wedge", "params": {"width": 10, "height": 10, "depth": 10}}
{"action": "cad", "command": "add_pyramid", "params": {"base": 10, "height": 15}}
```

### CAD Operations
```json
{"action": "cad", "command": "add_extrude", "params": {"height": 10}}
{"action": "cad", "command": "add_revolve", "params": {"angle": 360}}
{"action": "cad", "command": "add_fillet", "params": {"radius": 2}}
{"action": "cad", "command": "add_chamfer", "params": {"distance": 1}}
{"action": "cad", "command": "add_boolean", "params": {"operation": "union"}}
{"action": "cad", "command": "add_cut"}
{"action": "cad", "command": "add_shell", "params": {"thickness": 1}}
{"action": "cad", "command": "add_translate", "params": {"x": 10, "y": 0, "z": 0}}
{"action": "cad", "command": "add_rotate", "params": {"axis": "z", "angle": 45}}
{"action": "cad", "command": "add_mirror", "params": {"plane": "xy"}}
{"action": "cad", "command": "add_linear_pattern", "params": {"count": 5, "spacing": 10}}
{"action": "cad", "command": "add_circular_pattern", "params": {"count": 6, "angle": 360}}
```

### CAD Execution
```json
{"action": "cad", "command": "execute"}
{"action": "cad", "command": "export_stl", "params": {"filename": "model.stl"}}
{"action": "cad", "command": "export_step", "params": {"filename": "model.step"}}
```

### Project Commands
```json
{"action": "project", "command": "save"}
{"action": "project", "command": "new"}
{"action": "project", "command": "open"}
```

## Response Format

Always respond with a JSON block containing your actions, followed by a natural language explanation:

```json
{
  "thinking": "Brief explanation of your reasoning",
  "actions": [
    {"action": "...", "command": "...", "params": {...}, "description": "What this does"}
  ]
}
```

After the JSON, provide a friendly message explaining what you're doing.

## Examples

User: "Create a box with dimensions 50x30x20"
```json
{
  "thinking": "User wants a box primitive with specific dimensions",
  "actions": [
    {"action": "navigate", "command": "go_to_cad", "params": {}, "description": "Switch to CAD tab"},
    {"action": "cad", "command": "add_box", "params": {"width": 50, "height": 30, "depth": 20}, "description": "Create 50x30x20 box"}
  ]
}
```
I'll create a box with width=50, height=30, and depth=20 for you.

User: "Add an input called temperature with range 0 to 100"
```json
{
  "thinking": "User needs a design variable for temperature in modeling",
  "actions": [
    {"action": "navigate", "command": "go_to_modeling", "params": {}, "description": "Switch to Modeling tab"},
    {"action": "modeling", "command": "add_input", "params": {"name": "temperature", "min": 0, "max": 100}, "description": "Create temperature input"}
  ]
}
```
I'll add an input variable called "temperature" with a range of 0 to 100.

## Important Rules

1. Always include navigation commands if the user might not be on the right tab
2. Break complex operations into sequential steps
3. Use the description field to explain each action in plain language
4. If the user's request is unclear, ask for clarification instead of guessing
5. If you cannot perform an action, explain why and suggest alternatives
'''


class LLMInterpreter:
    """
    Interprets LLM responses and extracts executable actions.
    
    Parses JSON action blocks from LLM text and maps them to
    CommandDispatcher actions.
    """
    
    def __init__(self):
        """Initialize the interpreter."""
        # Get dynamic schema
        try:
            from pylcss.assistant_systems.utils.schema import get_simplified_schema_string
            self._node_schema = get_simplified_schema_string()
        except ImportError:
            logger.warning("Could not import node_schema_generator")
            self._node_schema = "Schema unavailable."
        except Exception as e:
            logger.error(f"Failed to generate node schema: {e}")
            self._node_schema = "Schema generation error."

        # Import ToolRegistry
        try:
            from pylcss.assistant_systems.tools.registry import ToolRegistry
            self._tool_registry = None # Registry provided by manager if needed
        except ImportError:
            self._tool_registry = None

        self._system_prompt = get_system_prompt() + "\n\n" + self._node_schema
        
        # Add Tools definition if available
        # Note: Manager will push tool updates if needed


        
        # Add build_node_graph capability to prompt
        self._system_prompt += '''
        
### ADVANCED: Batch Graph Construction
To create complex CAD models (like "Boeing 747" or "Car"), DO NOT add nodes one by one.
Instead, use `build_node_graph` to create the entire system at once.

### Example: Create a Simple Car
```json
{
  "action": "modeling", "command": "build_node_graph",
  "params": {
    "nodes": [
      {"id": "body", "type": "com.cad.box", "properties": {"width": 40, "height": 20, "depth": 10}},
      {"id": "cockpit", "type": "com.cad.box", "properties": {"width": 20, "height": 14, "depth": 10}},
      {"id": "cockpit_pos", "type": "com.cad.translate", "properties": {"z_translate": 10}},
      {"id": "wheel_FL", "type": "com.cad.cylinder", "properties": {"radius": 4, "height": 4}},
      {"id": "wheel_FL_rot", "type": "com.cad.rotate", "properties": {"axis": "x", "angle": 90}},
      {"id": "wheel_FL_pos", "type": "com.cad.translate", "properties": {"x_translate": 12, "y_translate": 12, "z_translate": -5}},
      {"id": "wheel_FR", "type": "com.cad.cylinder", "properties": {"radius": 4, "height": 4}},
      {"id": "wheel_FR_rot", "type": "com.cad.rotate", "properties": {"axis": "x", "angle": 90}},
      {"id": "wheel_FR_pos", "type": "com.cad.translate", "properties": {"x_translate": 12, "y_translate": -12, "z_translate": -5}},
      {"id": "wheel_BL", "type": "com.cad.cylinder", "properties": {"radius": 4, "height": 4}},
      {"id": "wheel_BL_rot", "type": "com.cad.rotate", "properties": {"axis": "x", "angle": 90}},
      {"id": "wheel_BL_pos", "type": "com.cad.translate", "properties": {"x_translate": -12, "y_translate": 12, "z_translate": -5}},
      {"id": "wheel_BR", "type": "com.cad.cylinder", "properties": {"radius": 4, "height": 4}},
      {"id": "wheel_BR_rot", "type": "com.cad.rotate", "properties": {"axis": "x", "angle": 90}},
      {"id": "wheel_BR_pos", "type": "com.cad.translate", "properties": {"x_translate": -12, "y_translate": -12, "z_translate": -5}},
      {"id": "union_body", "type": "com.cad.boolean", "properties": {"operation": "union"}}
    ],
    "connections": [
      {"from": "cockpit.shape", "to": "cockpit_pos.shape"},
      {"from": "body.shape", "to": "union_body.shape"},
      {"from": "cockpit_pos.shape", "to": "union_body.tool"},
      {"from": "wheel_FL.shape", "to": "wheel_FL_rot.shape"}, {"from": "wheel_FL_rot.shape", "to": "wheel_FL_pos.shape"},
      {"from": "wheel_FR.shape", "to": "wheel_FR_rot.shape"}, {"from": "wheel_FR_rot.shape", "to": "wheel_FR_pos.shape"},
      {"from": "wheel_BL.shape", "to": "wheel_BL_rot.shape"}, {"from": "wheel_BL_rot.shape", "to": "wheel_BL_pos.shape"},
      {"from": "wheel_BR.shape", "to": "wheel_BR_rot.shape"}, {"from": "wheel_BR_rot.shape", "to": "wheel_BR_pos.shape"}
    ]
  }
}
```

RULES for `build_node_graph`:
1. "id" can be any unique string.
2. "type" MUST be one of the identifiers listed in "Available CAD Nodes".
3. "properties" keys must match the available properties for that node type.
4. "connections" link "node_id.output_name" to "node_id.input_name".

### AGENTIC MODIFICATION
You may receive `[Current Graph State]` in the user message. This tells you what nodes currently exist.
To MODIFY an existing node (e.g. change dimensions):
1. Use `build_node_graph`.
2. In the "nodes" list, include an entry with the **SAME ID** as the existing node.
3. Provide the keys/values you want to UPDATE in "properties".
4. You do NOT need to list all properties, only the ones to change.

Example: Change 'box1' width to 100
```json
{
  "action": "modeling", "command": "build_node_graph",
  "params": {
    "nodes": [
      {"id": "box1", "properties": {"width": 100}}
    ]
  }
}
```

### Example: Create a Box with a Hole
User: "Create a box with a hole in the center"
```json
{
  "thinking": "User wants a box with a cylindrical hole cut out. I will use build_node_graph with Box, Cylinder, and Boolean(Cut).",
  "actions": [
    {
      "action": "modeling", 
      "command": "build_node_graph",
      "params": {
        "nodes": [
          {"id": "base_box", "type": "com.cad.box", "properties": {"box_length": 20, "box_width": 20, "box_depth": 5}},
          {"id": "hole_cyl", "type": "com.cad.cylinder", "properties": {"cyl_radius": 5, "cyl_height": 10}},
          {"id": "cut_op", "type": "com.cad.boolean", "properties": {"operation": "Cut"}}
        ],
        "connections": [
          {"from": "base_box.shape", "to": "cut_op.shape_a"},
          {"from": "hole_cyl.shape", "to": "cut_op.shape_b"}
        ]
      },
      "description": "Construct graph: Box - Cylinder = Hole"
    }
  ]
}
```
'''
        
        # Add build_system_graph capability
        self._system_prompt += '''

### ADVANCED: Batch System Graph Construction
To create complex System models, use `build_system_graph`.
Works exactly like `build_node_graph` but for the Modeling environment.
Node types: `com.pfd.input`, `com.pfd.output`, `com.pfd.intermediate`, `com.pfd.custom_block`.

Example: Create a simple f(x) = y system
```json
{
  "action": "modeling", "command": "build_system_graph",
  "params": {
    "nodes": [
      {"id": "in_var", "type": "com.pfd.input", "properties": {"var_name": "speed", "min": 0, "max": 100}},
      {"id": "engine", "type": "com.pfd.custom_block", "properties": {"code_content": "out_1 = in_1 * 2"}},
      {"id": "out_var", "type": "com.pfd.output", "properties": {"var_name": "power"}}
    ],
    "connections": [
      {"from": "in_var.x", "to": "engine.in_1"},
      {"from": "engine.out_1", "to": "out_var.y"}
    ]
  }
}
```
'''

        
        # Map LLM action types to dispatcher action types
        self._action_mapping = {
            # Navigation
            "navigate": {
                "go_to_modeling": {"action": "switch_tab", "tab": 0},
                "go_to_cad": {"action": "switch_tab", "tab": 1},
                "go_to_surrogate": {"action": "switch_tab", "tab": 2},
                "go_to_solution_space": {"action": "switch_tab", "tab": 3},
                "go_to_optimization": {"action": "switch_tab", "tab": 4},
                "go_to_sensitivity": {"action": "switch_tab", "tab": 5},
                "go_to_help": {"action": "switch_tab", "tab": 6},
            },
            # Modeling
            "modeling": {
                "add_input": {"action": "pylcss_action", "command": "add_input"},
                "add_output": {"action": "pylcss_action", "command": "add_output"},
                "add_function": {"action": "pylcss_action", "command": "add_function"},
                "add_intermediate": {"action": "pylcss_action", "command": "add_intermediate"},
                "add_system": {"action": "pylcss_action", "command": "add_system"},
                "validate_graph": {"action": "pylcss_action", "command": "validate_graph"},
                "build_model": {"action": "pylcss_action", "command": "build_model"},
                "clear_graph": {"action": "pylcss_action", "command": "clear_graph"},
                "auto_connect": {"action": "pylcss_action", "command": "auto_connect"},
                "build_node_graph": {"action": "pylcss_action", "command": "build_node_graph"},
                "build_system_graph": {"action": "pylcss_action", "command": "build_system_graph"},
                "train_surrogate": {"action": "pylcss_action", "command": "train_surrogate_node"},
                "connect_nodes": {"action": "pylcss_action", "command": "connect_nodes"},  # NEW
                "set_property": {"action": "pylcss_action", "command": "set_property"},    # NEW
            },
            # Optimization & Analysis
            "optimization": {
                 "get_sensitivity": {"action": "pylcss_action", "command": "get_sensitivity"}, # NEW
                 "run_optimization": {"action": "pylcss_action", "command": "run_optimization"},
            },

            # CAD
            "cad": {
                "add_box": {"action": "pylcss_action", "command": "cad_add_box"},
                "add_cylinder": {"action": "pylcss_action", "command": "cad_add_cylinder"},
                "add_sphere": {"action": "pylcss_action", "command": "cad_add_sphere"},
                "add_cone": {"action": "pylcss_action", "command": "cad_add_cone"},
                "add_torus": {"action": "pylcss_action", "command": "cad_add_torus"},
                "add_wedge": {"action": "pylcss_action", "command": "cad_add_wedge"},
                "add_pyramid": {"action": "pylcss_action", "command": "cad_add_pyramid"},
                "add_extrude": {"action": "pylcss_action", "command": "cad_add_extrude"},
                "add_revolve": {"action": "pylcss_action", "command": "cad_add_revolve"},
                "add_fillet": {"action": "pylcss_action", "command": "cad_add_fillet"},
                "add_chamfer": {"action": "pylcss_action", "command": "cad_add_chamfer"},
                "add_boolean": {"action": "pylcss_action", "command": "cad_add_boolean"},
                "add_cut": {"action": "pylcss_action", "command": "cad_add_cut"},
                "add_union": {"action": "pylcss_action", "command": "cad_add_union"},
                "add_shell": {"action": "pylcss_action", "command": "cad_add_shell"},
                "add_translate": {"action": "pylcss_action", "command": "cad_add_translate"},
                "add_rotate": {"action": "pylcss_action", "command": "cad_add_rotate"},
                "add_mirror": {"action": "pylcss_action", "command": "cad_add_mirror"},
                "add_scale": {"action": "pylcss_action", "command": "cad_add_scale"},
                "add_linear_pattern": {"action": "pylcss_action", "command": "cad_add_linear_pattern"},
                "add_circular_pattern": {"action": "pylcss_action", "command": "cad_add_circular_pattern"},
                "execute": {"action": "pylcss_action", "command": "cad_execute"},
                "export_stl": {"action": "pylcss_action", "command": "cad_export"},
                "export_step": {"action": "pylcss_action", "command": "cad_export_step"},
            },
            # Project
            "project": {
                "save": {"action": "keyboard", "keys": ["ctrl", "s"]},
                "new": {"action": "pylcss_action", "command": "new_project"},
                "open": {"action": "pylcss_action", "command": "open_project"},
            },
            # Tools
            "tool_use": {
                # This is a special catch-all pattern. 
                # The interpreter's action_to_dispatcher_format logic needs to handle dynamic tool names
                # strictly speaking, but for now we can map the known ones or modify that method.
                "create_graph": {"action": "tool_execution", "tool_name": "create_graph"},
                "execute_graph": {"action": "tool_execution", "tool_name": "execute_graph"},
            }
        }
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return self._system_prompt
        
    def _sanitize_json(self, json_str: str) -> str:
        """
        Sanitize JSON string by removing comments and trailing commas.
        """
        # Remove // comments
        json_str = re.sub(r'//.*', '', json_str)
        # Remove /* */ comments
        json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
        # Remove trailing commas before } or ]
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        return json_str

    def parse_response(self, response_text: str) -> ParsedResponse:
        """
        Parse an LLM response and extract actions.
        
        Args:
            response_text: Raw text from the LLM
            
        Returns:
            ParsedResponse with message and actions
        """
        actions: List[ActionCommand] = []
        thinking = ""
        message = response_text
        
        # Try to find JSON blocks in the response
        # Allow optional 'json' language identifier
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        json_matches = re.findall(json_pattern, response_text)
        
        # Also try to find bare JSON objects if no markdown blocks found
        if not json_matches:
            bare_json_pattern = r'\{\s*"(?:thinking|actions)"[\s\S]*?\}'
            bare_matches = re.findall(bare_json_pattern, response_text)
            json_matches.extend(bare_matches)
        
        for json_str in json_matches:
            try:
                # First try standard parse
                data = json.loads(json_str.strip())
            except json.JSONDecodeError:
                # Try sanitized parse
                try:
                    sanitized_str = self._sanitize_json(json_str)
                    data = json.loads(sanitized_str.strip())
                    logger.info("Successfully parsed JSON after sanitization.")
                except json.JSONDecodeError:
                    # Final fallback: Python literal eval (handling true/false/null)
                    try:
                        # Prepare string for python eval (json null -> None, but we want to map back)
                        # Actually we can just define the context or replace keywords
                        eval_str = json_str.replace("true", "True").replace("false", "False").replace("null", "None")
                        data = ast.literal_eval(eval_str)
                        logger.info("Successfully parsed JSON using AST fallback.")
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON block: {e}")
                        logger.debug(f"Bad JSON Content: {json_str}")
                        continue
            
            # Extract thinking
            if "thinking" in data:
                thinking = data["thinking"]
                
            # Extract actions
            if "actions" in data:
                for action_data in data["actions"]:
                    action = self._parse_action(action_data)
                    if action:
                        actions.append(action)

        # Remove JSON blocks from message
        message = re.sub(json_pattern, '', response_text).strip()
        message = re.sub(r'\{\s*"(?:thinking|actions)"[\s\S]*?\}', '', message).strip()
        
        return ParsedResponse(
            message=message,
            actions=actions,
            requires_confirmation=len(actions) > 0,
            thinking=thinking,
        )
        
    def _parse_action(self, action_data: Dict) -> Optional[ActionCommand]:
        """
        Parse a single action from JSON data.
        
        Args:
            action_data: Dictionary with action, command, params
            
        Returns:
            ActionCommand or None if invalid
        """
        action_type = action_data.get("action", "")
        command = action_data.get("command", "")
        params = action_data.get("params", {})
        description = action_data.get("description", "")
        
        if not action_type or not command:
            return None
            
        return ActionCommand(
            action_type=action_type,
            command=command,
            parameters=params,
            description=description,
        )
        
    def action_to_dispatcher_format(self, action: ActionCommand) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Convert an ActionCommand to CommandDispatcher format.
        
        Args:
            action: The ActionCommand to convert
            
        Returns:
            Tuple of (command_name, command_data) or None if not mappable
        """
        action_group = self._action_mapping.get(action.action_type, {})
        dispatcher_data = action_group.get(action.command)
        
        # Fallback for dynamic tools if not explicitly mapped
        if not dispatcher_data and action.action_type == "tool_use":
             dispatcher_data = {"action": "tool_execution", "tool_name": action.command}
        
        if not dispatcher_data:
            logger.warning(f"Unknown action: {action.action_type}.{action.command}")
            return None
            
        # Create command data with parameters
        command_data = dispatcher_data.copy()
        command_data["params"] = action.parameters
        
        # Generate command name from action
        command_name = f"{action.action_type}_{action.command}"
        
        return (command_name, command_data)
        
    def get_action_summary(self, actions: List[ActionCommand]) -> str:
        """
        Generate a human-readable summary of actions.
        
        Args:
            actions: List of actions to summarize
            
        Returns:
            Summary string
        """
        if not actions:
            return "No actions to perform."
            
        lines = []
        for i, action in enumerate(actions, 1):
            desc = action.description or f"{action.action_type}: {action.command}"
            lines.append(f"{i}. {desc}")
            
        return "\n".join(lines)
