# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
LLM Command Interpreter for PyLCSS.

Parses LLM responses and translates them into executable commands
for both the Modeling environment and Design Studio.
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
    return '''You are the engineering assistant for PyLCSS.

Domains:
- modeling: inputs, outputs, intermediate variables, custom blocks, validation, model build
- cad: parametric geometry, booleans, transforms, patterns, execution, export
- optimization: sensitivity and optimization actions
- navigation/project: tab switching, save, open, new

Respond with compact JSON followed by a short plain-language message.

```json
{
  "thinking": "brief reasoning",
  "actions": [
    {"action": "navigate|modeling|cad|optimization|project", "command": "...", "params": {}, "description": "..."}
  ]
}
```

Rules:
1. Include navigation when the tab may be wrong.
2. Prefer batch graph actions for multi-node tasks.
3. Keep descriptions short.
4. Ask for clarification if the request is ambiguous.
5. If unsupported, explain briefly.
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


        self._system_prompt += '''

### Batch Graph Commands
- Use `build_node_graph` for multi-node CAD creation or edits.
- Use `build_system_graph` for multi-node system-model creation or edits.
- Node entries use `{id, type, properties}`.
- Connection entries use `{"from": "node.port", "to": "node.port"}`.
- To modify existing nodes from `[Current Graph State]`, reuse the same `id` and include only changed properties.
- Prefer one batch action instead of many small node actions when building a graph.
'''

        self._system_prompt += '''

### System Modeling Notes
- System node types are `com.pfd.input`, `com.pfd.output`, `com.pfd.intermediate`, and `com.pfd.custom_block`.
- Input/output/intermediate ports use `var_name`.
- Custom block ports use `in_1..in_n` and `out_1..out_n`.
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
