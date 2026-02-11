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
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import logging
import re


if TYPE_CHECKING:
    from pylcss.assistant_systems.api.dispatcher import CommandDispatcher

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
    
    def initialize_defaults(self):
        """Initialize default tools."""
        # This method is called by the dispatcher to ensure default tools are loaded
        # Note: In the current architecture, create_pylcss_tools handles registration,
        # but this method is kept for compatibility if needed.
        pass

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

    # -- validation helpers ---------------------------------------------------

    def _sanitize_cad_params(data: Dict) -> Dict:
        """Strip hallucinated node types and properties before dispatch.

        The LLM sometimes invents properties that don't exist on a node.
        Rather than letting the dispatcher silently ignore them (or crash),
        we log a warning and remove them so the rest of the graph still
        works.  Unknown node types are also flagged.
        """
        # Sketch-element types that NEED the 'sketch.' sub-prefix:
        #   sketch.circle, sketch.rectangle, sketch.polygon, sketch.arc, sketch.line
        # Sketch-element types WITHOUT 'sketch.' sub-prefix:
        #   polyline, spline, ellipse
        _SHORT_FIXES = {
            # LLM drops 'sketch.' prefix for types that need it
            "circle": "sketch.circle",
            "rectangle": "sketch.rectangle",
            "polygon": "sketch.polygon",
            "arc": "sketch.arc",
            "line": "sketch.line",
            # LLM wrongly adds 'sketch.' prefix for types that don't need it
            "sketch.polyline": "polyline",
            "sketch.spline": "spline",
            "sketch.ellipse": "ellipse",
            # Underscore instead of dot
            "sketch_circle": "sketch.circle",
            "sketch_rectangle": "sketch.rectangle",
            "sketch_polygon": "sketch.polygon",
            "sketch_arc": "sketch.arc",
            "sketch_line": "sketch.line",
            "sketch_polyline": "polyline",
            "sketch_spline": "spline",
            "sketch_ellipse": "ellipse",
        }

        # Strip hallucinated proxy nodes (e.g. type="parameter") that the
        # LLM creates when trying to reference earlier sub-part outputs.
        # These nodes don't exist — the connections should go directly to
        # the earlier node IDs that already live in the graph.
        _PHANTOM_TYPES = {"com.cad.parameter", "com.cad.reference", "com.cad.proxy",
                          "com.cad.input", "com.cad.output", "com.cad.ref"}
        nodes = data.get("nodes", [])
        conns = data.get("connections", [])

        # Identify phantom nodes and build a bypass map:
        # phantom_id.port  →  the upstream node.port that feeds into it.
        phantom_ids: set = set()
        bypass_map: Dict[str, str] = {}   # "phantom.shape" → "real_upstream.shape"
        pre_nodes = []
        for node in nodes:
            ntype = node.get("type", "")
            if ntype and not ntype.startswith("com.cad."):
                ntype = "com.cad." + ntype
            if ntype in _PHANTOM_TYPES:
                phantom_ids.add(node.get("id", ""))
            else:
                pre_nodes.append(node)

        if phantom_ids:
            # For every connection that feeds INTO a phantom node, record
            # where the phantom's output should actually come from.
            for c in conns:
                from_str, to_str = c.get("from", ""), c.get("to", "")
                if "." in to_str:
                    tid = to_str.split(".", 1)[0]
                    if tid in phantom_ids:
                        # The phantom's output port = whatever fed into it
                        phantom_out = f"{tid}.shape"
                        bypass_map[phantom_out] = from_str

            # Rewrite connections: remove those touching phantoms; redirect
            # downstream connections through the bypass map.
            new_conns = []
            for c in conns:
                from_str, to_str = c.get("from", ""), c.get("to", "")
                fid = from_str.split(".", 1)[0] if "." in from_str else ""
                tid = to_str.split(".", 1)[0] if "." in to_str else ""
                if fid in phantom_ids or tid in phantom_ids:
                    continue  # drop connections involving the phantom itself
                new_conns.append(c)

            # For connections that referenced a phantom's output, swap in
            # the real upstream source.
            for c in new_conns:
                from_str = c.get("from", "")
                if from_str in bypass_map:
                    c["from"] = bypass_map[from_str]

            dropped = len(nodes) - len(pre_nodes)
            logger.info(
                f"Stripped {dropped} phantom proxy node(s): "
                f"{phantom_ids}; bypass map: {bypass_map}"
            )
            data["nodes"] = pre_nodes
            data["connections"] = new_conns
            nodes = pre_nodes
            conns = new_conns

        for node in nodes:
            ntype = node.get("type", "")

            # Unwrap nested 'custom' properties — the LLM sometimes
            # parrots the graph-state format which nests real properties
            # inside a 'custom' dict alongside internal keys.
            props = node.get("properties", {})
            if "custom" in props and isinstance(props["custom"], dict):
                # Pull custom values up, drop internal keys
                custom = props.pop("custom")
                internal_keys = {"type_", "name", "visible", "layout_direction",
                                 "subgraph_session", "selected", "disabled",
                                 "id", "icon"}
                for k in list(props.keys()):
                    if k in internal_keys:
                        del props[k]
                props.update(custom)
                node["properties"] = props
                logger.info(f"Unwrapped nested 'custom' properties for node '{node.get('id', '?')}'")

            # Strip class-name suffix: com.cad.box.BoxNode → com.cad.box
            # The LLM sometimes copies full class paths from the graph state.
            if ntype and re.search(r'\.[A-Z][a-zA-Z]*Node$', ntype):
                ntype = re.sub(r'\.[A-Z][a-zA-Z]*Node$', '', ntype)
                node["type"] = ntype
                logger.info(f"Stripped class suffix from type → {ntype}")

            # Auto-prepend com.cad. if the LLM omitted it
            if ntype and not ntype.startswith("com.cad."):
                ntype = "com.cad." + ntype
                node["type"] = ntype

            # Strip prefix for lookup in short fixes table
            short = ntype.replace("com.cad.", "", 1)
            if short in _SHORT_FIXES:
                fixed_short = _SHORT_FIXES[short]
                fixed = "com.cad." + fixed_short
                logger.info(f"Auto-corrected node type: {ntype} → {fixed}")
                node["type"] = fixed
                ntype = fixed

            schema = CAD_NODE_TYPES.get(ntype)
            if not schema:
                logger.warning(f"LLM used unknown CAD node type: {ntype}")
                continue
            valid_props = set(schema.get("properties", {}).keys())
            if not valid_props:
                continue
            props = node.get("properties", {})

            # --- Remap commonly hallucinated property names ----------------
            # The LLM frequently invents synonyms for real properties.
            # Map them to the correct names BEFORE the stripping step.
            _PROP_ALIASES: Dict[str, Dict[str, str]] = {
                "com.cad.twisted_extrude": {
                    "extrude_distance": "distance",
                    "twist_angle": "angle",
                    "helix_angle": "angle",
                    "twist": "angle",
                    "height": "distance",
                    "length": "distance",
                },
                "com.cad.extrude": {
                    "distance": "extrude_distance",
                    "height": "extrude_distance",
                    "length": "extrude_distance",
                },
                "com.cad.cut_extrude": {
                    "extrude_distance": "distance",
                    "height": "distance",
                    "length": "distance",
                    "depth": "distance",
                },
                "com.cad.chamfer": {
                    "chamfer_distance": "distance",
                    "size": "distance",
                },
                "com.cad.fillet": {
                    "radius": "fillet_radius",
                    "fillet_size": "fillet_radius",
                },
                "com.cad.revolve": {
                    "revolve_angle": "angle",
                    "rotation_angle": "angle",
                },
            }
            alias_table = _PROP_ALIASES.get(ntype, {})
            if alias_table:
                remapped = []
                for bad_name, good_name in alias_table.items():
                    if bad_name in props and good_name not in props:
                        props[good_name] = props.pop(bad_name)
                        remapped.append(f"{bad_name}→{good_name}")
                if remapped:
                    logger.info(
                        f"Remapped hallucinated props for {ntype}: "
                        f"{', '.join(remapped)}"
                    )

            bad = [k for k in props if k not in valid_props
                   and k not in ("center_x", "center_y", "center_z")]  # center_* common to all
            if bad:
                logger.warning(
                    f"Stripping hallucinated props from {ntype}: {bad}  "
                    f"(valid: {sorted(valid_props)})"
                )
                for k in bad:
                    del props[k]

        # --- Fuzzy-repair connection references ----------------------------
        # The LLM frequently makes small typos in node IDs inside connections
        # (e.g. 'p1_sk_poly1' instead of 'p1_sk1_poly1').  Walk every
        # connection endpoint; if the referenced node-ID doesn't exist, find
        # the most similar real node-ID and fix the reference.
        # SKIP cross-references to earlier sub-parts (different pN_ prefix)
        # — those are resolved at dispatch time by the existing graph.
        node_ids = {n.get("id", "") for n in nodes}
        local_prefixes = set()
        for nid in node_ids:
            m = re.match(r"(p\d+_)", nid)
            if m:
                local_prefixes.add(m.group(1))

        conns = data.get("connections", [])
        for conn in conns:
            for key in ("from", "to"):
                ref = conn.get(key, "")
                if "." not in ref:
                    continue
                ref_id, port = ref.split(".", 1)
                if ref_id in node_ids:
                    continue  # already correct
                # Skip cross-references to earlier sub-parts
                m = re.match(r"(p\d+_)", ref_id)
                if m and m.group(1) not in local_prefixes:
                    continue  # valid cross-reference, will be resolved by dispatcher
                # Find the closest matching node ID
                best, best_score = None, 0.0
                for nid in node_ids:
                    score = SequenceMatcher(None, ref_id, nid).ratio()
                    if score > best_score:
                        best, best_score = nid, score
                if best and best_score >= 0.7:
                    conn[key] = f"{best}.{port}"
                    logger.info(
                        f"Auto-fixed connection ID typo: '{ref_id}' → '{best}' "
                        f"(similarity {best_score:.0%})"
                    )
                else:
                    logger.warning(
                        f"Connection references unknown node '{ref_id}' with "
                        f"no close match (best: '{best}' at {best_score:.0%})"
                    )

        return data

    def _verify_cad_graph(data: Dict) -> List[str]:
        """Verify structural integrity of an LLM-generated CAD node graph.

        Returns a list of issue strings (empty = all good).
        Checks:
        1. Disconnected nodes (no connections at all)
        2. Boolean nodes missing one or both inputs
        3. Unknown node types
        4. All properties still at default (nothing customized)
        5. Connections referencing non-existent node IDs
        6. Duplicate node IDs
        """
        nodes = data.get("nodes", [])
        conns = data.get("connections", [])
        issues: List[str] = []

        if not nodes:
            return ["Empty node list — nothing to build"]

        node_ids = set()
        node_map: Dict[str, Dict] = {}
        for n in nodes:
            nid = n.get("id", "")
            if nid in node_ids:
                issues.append(f"Duplicate node ID: '{nid}'")
            node_ids.add(nid)
            node_map[nid] = n

        # Nodes mentioned in connections
        connected_ids: set = set()
        # Detect the sub-part prefix of *this* call's nodes (e.g. "p8_")
        # so we can distinguish cross-references to earlier sub-parts from
        # genuine typos.
        local_prefixes = set()
        for nid in node_ids:
            m = re.match(r"(p\d+_)", nid)
            if m:
                local_prefixes.add(m.group(1))

        for c in conns:
            from_str = c.get("from", "")
            to_str = c.get("to", "")
            if "." in from_str:
                fid = from_str.split(".", 1)[0]
                connected_ids.add(fid)
                if fid not in node_ids:
                    # Check if it looks like a cross-reference to an
                    # earlier sub-part (different pN_ prefix).  Those are
                    # resolved at dispatch time by the existing graph.
                    m = re.match(r"(p\d+_)", fid)
                    if m and m.group(1) not in local_prefixes:
                        pass  # valid cross-reference — skip warning
                    else:
                        issues.append(f"Connection references unknown node: '{fid}'")
            if "." in to_str:
                tid = to_str.split(".", 1)[0]
                connected_ids.add(tid)
                if tid not in node_ids:
                    m = re.match(r"(p\d+_)", tid)
                    if m and m.group(1) not in local_prefixes:
                        pass  # valid cross-reference
                    else:
                        issues.append(f"Connection references unknown node: '{tid}'")

        # Check each node
        for n in nodes:
            nid = n.get("id", "?")
            ntype = n.get("type", "")
            props = n.get("properties", {})

            # Unknown type
            schema = CAD_NODE_TYPES.get(ntype)
            if not schema:
                issues.append(f"Node '{nid}': unknown type '{ntype}'")
                continue

            # Disconnected — node has inputs defined but nothing connects to it
            schema_inputs = schema.get("inputs", [])
            if schema_inputs and nid not in connected_ids:
                issues.append(
                    f"Node '{nid}' ({ntype}) is disconnected — "
                    f"has inputs {schema_inputs} but no connections"
                )

            # Boolean missing connections
            if ntype == "com.cad.boolean":
                a_connected = any(
                    c.get("to", "").startswith(f"{nid}.shape_a") for c in conns
                )
                b_connected = any(
                    c.get("to", "").startswith(f"{nid}.shape_b") for c in conns
                )
                if not a_connected:
                    issues.append(f"Boolean '{nid}' missing shape_a connection")
                if not b_connected:
                    issues.append(f"Boolean '{nid}' missing shape_b connection")

            # Numeric properties all at default — LLM didn't customize dims
            defaults = schema.get("properties", {})
            if defaults and props:
                # Only check numeric properties (dimensions, counts, angles).
                # String/enum props (like operation='Union') are intentional.
                # Skip nodes that have NO numeric defaults (e.g. loft with only 'ruled').
                numeric_defaults = {
                    k: v for k, v in defaults.items()
                    if isinstance(v, (int, float)) and k in props
                }
                if len(numeric_defaults) >= 2:
                    all_default = all(
                        props.get(k) == v or str(props.get(k)) == str(v)
                        for k, v in numeric_defaults.items()
                    )
                    if all_default:
                        issues.append(
                            f"Node '{nid}' ({ntype}): all numeric properties at "
                            f"defaults {numeric_defaults} — LLM may not have "
                            f"customized dimensions"
                        )

        return issues

    def _verify_system_graph(data: Dict) -> List[str]:
        """Verify structural integrity of an LLM-generated system modeling graph.

        Returns a list of issue strings (empty = all good).
        Checks:
        1. Disconnected nodes
        2. CustomBlock with empty/commented-out code_content
        3. Input nodes with min >= max
        4. Output node with both minimize and maximize
        5. Connections referencing non-existent nodes
        """
        nodes = data.get("nodes", [])
        conns = data.get("connections", [])
        issues: List[str] = []

        if not nodes:
            return ["Empty node list"]

        node_ids = set()
        node_map: Dict[str, Dict] = {}
        for n in nodes:
            nid = n.get("id", "")
            node_ids.add(nid)
            node_map[nid] = n

        connected_ids: set = set()
        for c in conns:
            for key in ("from", "to"):
                s = c.get(key, "")
                if "." in s:
                    nid = s.split(".", 1)[0]
                    connected_ids.add(nid)
                    if nid not in node_ids:
                        issues.append(f"Connection references unknown node: '{nid}'")

        for n in nodes:
            nid = n.get("id", "?")
            ntype = n.get("type", "")
            props = n.get("properties", {})

            # Disconnected input (should connect to something)
            if ntype == "com.pfd.input" and nid not in connected_ids:
                issues.append(f"Input '{nid}' is disconnected — not wired to any function")

            # Disconnected output
            if ntype == "com.pfd.output" and nid not in connected_ids:
                issues.append(f"Output '{nid}' is disconnected — nothing feeds into it")

            # CustomBlock with empty code
            if ntype == "com.pfd.custom_block":
                code = str(props.get("code_content", ""))
                # Strip comments and whitespace
                real_code = "\n".join(
                    line for line in code.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                )
                if not real_code.strip():
                    issues.append(
                        f"CustomBlock '{nid}' has empty/commented-out code_content"
                    )

            # Input min >= max
            if ntype == "com.pfd.input":
                try:
                    mn = float(props.get("min", 0))
                    mx = float(props.get("max", 10))
                    if mn >= mx:
                        issues.append(f"Input '{nid}': min ({mn}) >= max ({mx})")
                except (ValueError, TypeError):
                    pass

            # Output with both minimize and maximize
            if ntype == "com.pfd.output":
                if props.get("minimize") and props.get("maximize"):
                    issues.append(
                        f"Output '{nid}': both minimize and maximize set — pick one"
                    )

        return issues

    def _run_cad_verified(data: Dict) -> Any:
        """Sanitize, verify, then dispatch CAD graph. Returns issues if critical."""
        data = _sanitize_cad_params(data)
        issues = _verify_cad_graph(data)
        if issues:
            for issue in issues:
                logger.warning(f"CAD graph issue: {issue}")
        # Dispatch even with warnings — let the engine handle what it can
        result = command_dispatcher._build_node_graph({"params": data}, sync=True)
        if issues:
            return f"{result or 'Graph created'} | ⚠ Verifier warnings: {'; '.join(issues)}"
        return result

    def _run_system_verified(data: Dict) -> Any:
        """Verify then dispatch system modeling graph. Returns issues if critical."""
        issues = _verify_system_graph(data)
        if issues:
            for issue in issues:
                logger.warning(f"System graph issue: {issue}")
        result = command_dispatcher._build_system_graph({"params": data}, sync=True)
        if issues:
            return f"{result or 'Graph created'} | ⚠ Verifier warnings: {'; '.join(issues)}"
        return result

    # === CAD Tools ===

    registry.register(Tool(
        name="create_cad_geometry",
        description="Create CAD geometry using a node graph. \n\nExamples:\n- Basic: `nodes=[{'type': 'com.cad.box', 'id':'b1'}]`\n- Connection: `connections=[{'from': 'b1.shape', 'to': 'bool1.shape_a'}]`\n- Drilling: Create Box AND Cylinder. Then use `com.cad.boolean` (operation='Cut') and connect them.",
        parameters=[
            ToolParameter("nodes", "array", "List of nodes. Each node has: id (string), type (e.g. 'com.cad.box'), properties (object)"),
            ToolParameter("connections", "array", "Connections list. Each has: from (node_id.output_port), to (node_id.input_port)", required=False),
        ],
        handler=lambda data: _run_cad_verified(data),
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

    
    # === Modeling Tools ===
    
    registry.register(Tool(
        name="create_system_model",
        description=(
            "Create a system model with inputs, outputs, functions, and intermediates.\n"
            "Node types: com.pfd.input (Design Variable), com.pfd.output (QoI), "
            "com.pfd.intermediate (pass-through), com.pfd.custom_block (Python function).\n"
            "Port naming: Input/Output/Intermediate ports = var_name. CustomBlock ports = in_1, in_2, ... / out_1, out_2, ...\n"
            "Connection format: {\"from\": \"node_id.port_name\", \"to\": \"node_id.port_name\"}\n"
            "Example: {\"from\": \"width.width\", \"to\": \"calc.in_1\"} connects InputNode 'width' to CustomBlock 'calc'"
        ),
        parameters=[
            ToolParameter("nodes", "array",
                          "List of system nodes. Each: {id, type, properties}. "
                          "Input props: var_name, unit, min, max. "
                          "Output props: var_name, unit, req_min, req_max, minimize, maximize. "
                          "Intermediate props: var_name, unit. "
                          "CustomBlock props: num_inputs, num_outputs, code_content."),
            ToolParameter("connections", "array",
                          "Connections: [{from: 'nodeId.portName', to: 'nodeId.portName'}]. "
                          "Use var_name as port for I/O/Intermediate. Use in_1/out_1 for CustomBlock.",
                          required=False),
        ],
        handler=lambda data: _run_system_verified(data),
        category="modeling",
    ))
    
    registry.register(Tool(
        name="add_input_variable",
        description="Add a design variable (input) to the system model. Output port is named after var_name.",
        parameters=[
            ToolParameter("name", "string", "Variable name (becomes the output port name)"),
            ToolParameter("min", "number", "Minimum value in the design space"),
            ToolParameter("max", "number", "Maximum value in the design space"),
            ToolParameter("unit", "string", "Physical unit (pint-compatible, e.g. 'mm', 'kg', 'N/m^2', '-' for dimensionless)", required=False, default="-"),
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
                        "unit": data.get("unit", "-"),
                    }
                }]
            }
        }, sync=True),
        category="modeling",

    ))
    
    registry.register(Tool(
        name="modify_system_node",
        description="Modify properties of a system node (names, min/max, code_content).",
        parameters=[
            ToolParameter("node_id", "string", "ID/Name of the node to modify"),
            ToolParameter("properties", "object", "Map of properties to update (e.g. {'var_name': 'x2', 'min': 5, 'code_content': '# code...'})"),
        ],
        handler=lambda data: command_dispatcher._modify_system_node({"params": data}, sync=True),
        category="modeling",
    ))
    
    registry.register(Tool(
        name="get_graph_state",
        description="Get the current state of the graph (nodes, IDs, properties). Use this before editing to find node IDs.",
        parameters=[],
        handler=lambda data: command_dispatcher._get_graph_state(sync=True),
        category="modeling",
    ))
    
    registry.register(Tool(
        name="add_output_variable",
        description="Add an output (Quantity of Interest / QoI) to the system model. "
                    "Set requirements with req_min/req_max. Set minimize=True or maximize=True for optimization objectives. "
                    "Input port is named after var_name.",
        parameters=[
            ToolParameter("name", "string", "Output variable name (becomes the input port name)"),
            ToolParameter("unit", "string", "Physical unit (pint-compatible, e.g. 'mm', 'kg', '-')", required=False, default="-"),
            ToolParameter("req_min", "number", "Requirement lower bound (use -1e9 for unconstrained)", required=False),
            ToolParameter("req_max", "number", "Requirement upper bound (use 1e9 for unconstrained)", required=False),
            ToolParameter("minimize", "boolean", "Set True to minimize this output (optimization objective)", required=False, default=False),
            ToolParameter("maximize", "boolean", "Set True to maximize this output (optimization objective)", required=False, default=False),
        ],
        handler=lambda data: command_dispatcher._build_system_graph({
            "params": {
                "nodes": [{
                    "id": data.get("name", "output"),
                    "type": "com.pfd.output",
                    "properties": {
                        k: v for k, v in {
                            "var_name": data.get("name"),
                            "unit": data.get("unit", "-"),
                            "req_min": data.get("req_min"),
                            "req_max": data.get("req_max"),
                            "minimize": data.get("minimize"),
                            "maximize": data.get("maximize"),
                        }.items() if v is not None
                    }
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
# Property names MUST match the exact create_property names in the node classes.

CAD_NODE_TYPES = {
    # --- Primitives (all output: shape) ---
    "com.cad.box": {
        "name": "Box",
        "properties": {"box_length": 10.0, "box_width": 10.0, "box_depth": 10.0,
                        "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
        "outputs": ["shape"],
    },
    "com.cad.cylinder": {
        "name": "Cylinder",
        "properties": {"cyl_radius": 5.0, "cyl_height": 10.0,
                        "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
        "outputs": ["shape"],
    },
    "com.cad.sphere": {
        "name": "Sphere",
        "properties": {"sphere_radius": 5.0, "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
        "outputs": ["shape"],
    },
    "com.cad.cone": {
        "name": "Cone",
        "properties": {"bottom_radius": 10.0, "top_radius": 5.0, "cone_height": 20.0,
                        "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
        "outputs": ["shape"],
    },
    "com.cad.torus": {
        "name": "Torus",
        "properties": {"major_radius": 20.0, "minor_radius": 5.0,
                        "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
        "outputs": ["shape"],
    },
    "com.cad.wedge": {
        "name": "Wedge",
        "properties": {"wedge_width": 10.0, "length": 10.0, "wedge_height": 5.0,
                        "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
        "outputs": ["shape"],
    },
    "com.cad.pyramid": {
        "name": "Pyramid",
        "properties": {"base_size": 10.0, "pyramid_height": 15.0, "sides": 4,
                        "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
        "outputs": ["shape"],
    },

    # --- Sketching ---
    "com.cad.sketch": {
        "name": "Create Sketch",
        "properties": {"sketch_name": "Sketch1", "plane": "XY"},
        "outputs": ["sketch"],
    },
    "com.cad.sketch.line": {
        "name": "Line",
        "properties": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
        "inputs": ["sketch"], "outputs": ["shape"],
    },
    "com.cad.sketch.circle": {
        "name": "Circle",
        "properties": {"center_x": 0.0, "center_y": 0.0, "radius": 5.0},
        "inputs": ["sketch"], "outputs": ["shape"],
    },
    "com.cad.sketch.rectangle": {
        "name": "Rectangle",
        "properties": {"x": 0.0, "y": 0.0, "rect_width": 10.0, "rect_height": 10.0},
        "inputs": ["sketch"], "outputs": ["shape"],
    },
    "com.cad.sketch.polygon": {
        "name": "Polygon",
        "properties": {"center_x": 0.0, "center_y": 0.0, "radius": 5.0, "sides": 6},
        "inputs": ["sketch"], "outputs": ["shape"],
    },
    "com.cad.sketch.arc": {
        "name": "Arc",
        "properties": {"center_x": 0.0, "center_y": 0.0, "radius": 10.0,
                        "start_angle": 0.0, "end_angle": 90.0},
        "inputs": ["sketch"], "outputs": ["shape"],
    },
    "com.cad.ellipse": {
        "name": "Ellipse",
        "properties": {"x_radius": 10.0, "y_radius": 5.0, "center_x": 0.0, "center_y": 0.0},
        "inputs": ["sketch"], "outputs": ["shape"],
    },
    "com.cad.spline": {
        "name": "Spline",
        "properties": {"points": "[(0,0), (5,5), (10,0), (15,5)]"},
        "inputs": ["sketch"], "outputs": ["shape"],
    },
    "com.cad.polyline": {
        "name": "Polyline",
        "properties": {"points": "[(0,0), (5,5), (5,0)]", "closed": True},
        "inputs": ["sketch"], "outputs": ["shape"],
    },

    # --- 3D Operations ---
    "com.cad.extrude": {
        "name": "Extrude",
        "properties": {"extrude_distance": 5.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.revolve": {
        "name": "Revolve",
        "properties": {"angle": 360.0, "axis_start_x": 0.0, "axis_start_y": 0.0,
                        "axis_start_z": 0.0, "axis_dir_x": 0.0, "axis_dir_y": 1.0, "axis_dir_z": 0.0},
        "inputs": ["profile"], "outputs": ["shape"],
    },
    "com.cad.twisted_extrude": {
        "name": "Twisted Extrude",
        "properties": {"distance": 10.0, "angle": 45.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.cut_extrude": {
        "name": "Extruded Cut",
        "properties": {"distance": 10.0, "through_all": False},
        "inputs": ["target", "profile"], "outputs": ["shape"],
    },
    "com.cad.boolean": {
        "name": "Boolean Op",
        "properties": {"operation": "Union"},  # Union / Cut / Intersect
        "inputs": ["shape_a", "shape_b"], "outputs": ["shape"],
    },
    "com.cad.fillet": {
        "name": "Fillet",
        "properties": {"fillet_radius": 1.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.chamfer": {
        "name": "Chamfer",
        "properties": {"distance": 1.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.shell": {
        "name": "Shell",
        "properties": {"thickness": -2.0},
        "inputs": ["shape", "face_to_remove"], "outputs": ["shape"],
    },
    "com.cad.sweep": {
        "name": "Sweep",
        "properties": {},
        "inputs": ["profile", "path"], "outputs": ["shape"],
    },
    "com.cad.loft": {
        "name": "Loft",
        "properties": {"ruled": True},
        "inputs": ["profiles"], "outputs": ["shape"],
    },
    "com.cad.select_face": {
        "name": "Select Face",
        "properties": {"selector_type": "Direction", "direction": ">Z"},
        "inputs": ["shape"], "outputs": ["workplane"],
    },

    # --- Cutting / Holes ---
    "com.cad.hole_at_coords": {
        "name": "Hole (Point)",
        "properties": {"x_position": 10.0, "y_position": 10.0, "diameter": 5.0,
                        "depth": 10.0, "through_all": True, "from_face": ">Z"},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.multi_hole": {
        "name": "Holes (Multi)",
        "properties": {"coordinates": "[(10,10), (40,10)]", "diameter": 5.0,
                        "depth": 10.0, "through_all": True, "from_face": ">Z"},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.pocket": {
        "name": "Pocket",
        "properties": {"pocket_depth": 5.0, "pocket_radius": 2.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.rectangular_cut": {
        "name": "Rectangular Pocket",
        "properties": {"x_position": 10.0, "y_position": 10.0, "cut_width": 15.0,
                        "cut_length": 20.0, "depth": 5.0, "through_all": False, "from_face": ">Z"},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.slot_cut": {
        "name": "Slot",
        "properties": {"x_start": 10.0, "y_start": 10.0, "x_end": 30.0, "y_end": 10.0,
                        "slot_width": 5.0, "depth": 5.0, "through_all": False, "from_face": ">Z"},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.array_holes": {
        "name": "Hole Array",
        "properties": {"x_start": 10.0, "y_start": 10.0, "x_spacing": 20.0, "y_spacing": 15.0,
                        "x_count": 3, "y_count": 2, "diameter": 5.0, "through_all": True, "from_face": ">Z"},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.cylinder_cut": {
        "name": "Cylinder Cut",
        "properties": {"cyl_radius": 5.0, "cyl_height": 10.0,
                        "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
        "inputs": ["target"], "outputs": ["shape"],
    },

    # --- Transforms ---
    "com.cad.translate": {
        "name": "Translate",
        "properties": {"x": 10.0, "y": 0.0, "z": 0.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.rotate": {
        "name": "Rotate",
        "properties": {"angle": 45.0, "axis_x": 0.0, "axis_y": 0.0, "axis_z": 1.0,
                        "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.scale": {
        "name": "Scale",
        "properties": {"factor": 2.0, "uniform": True,
                        "x_factor": 1.0, "y_factor": 1.0, "z_factor": 1.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.mirror": {
        "name": "Mirror",
        "properties": {"plane": "XY", "union": False},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.offset": {
        "name": "Offset 2D",
        "properties": {"distance": 2.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },

    # --- Patterns ---
    "com.cad.linear_pattern": {
        "name": "Linear Pattern",
        "properties": {"count": 3, "distance": 10.0, "dx": 1.0, "dy": 0.0, "dz": 0.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.circular_pattern": {
        "name": "Circular Pattern",
        "properties": {"count": 6, "angle": 360.0, "axis_x": 0.0, "axis_y": 0.0, "axis_z": 1.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.pattern.radial": {
        "name": "Radial Pattern",
        "properties": {"count": 6, "angle": 360.0,
                        "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.pattern.mirror": {
        "name": "Mirror Pattern",
        "properties": {"plane_normal_x": 1.0, "plane_normal_y": 0.0, "plane_normal_z": 0.0,
                        "plane_point_x": 0.0, "plane_point_y": 0.0, "plane_point_z": 0.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },
    "com.cad.pattern.grid": {
        "name": "Grid Pattern",
        "properties": {"rows": 3, "columns": 3, "row_spacing": 10.0, "col_spacing": 10.0},
        "inputs": ["shape"], "outputs": ["shape"],
    },

    # --- Assembly ---
    "com.cad.assembly": {
        "name": "Assembly",
        "properties": {"assembly_name": "Assembly1"},
        "inputs": ["part_1", "part_2", "part_3", "part_4"], "outputs": ["assembly"],
    },

    # --- Analysis ---
    "com.cad.mass_properties": {
        "name": "Mass Properties",
        "properties": {"density": 7850.0},
        "inputs": ["shape"], "outputs": ["properties"],
    },
    "com.cad.bounding_box": {
        "name": "Bounding Box",
        "properties": {},
        "inputs": ["shape"], "outputs": ["dimensions"],
    },

    # --- FEA Simulation ---
    "com.cad.sim.material": {
        "name": "Material",
        "properties": {"preset": "Steel (Structural)", "youngs_modulus": 210000.0,
                        "poissons_ratio": 0.3, "density": 7.85e-9},
        "outputs": ["material"],
    },
    "com.cad.sim.mesh": {
        "name": "Generate Mesh",
        "properties": {"mesh_type": "Tet", "element_size": 2.0, "refinement_size": 0.5},
        "inputs": ["shape"], "outputs": ["mesh"],
    },
    "com.cad.sim.constraint": {
        "name": "FEA Constraint",
        "properties": {"constraint_type": "Fixed"},  # Fixed/Roller X/Y/Z/Pinned/Symmetry/Displacement
        "inputs": ["mesh", "target_face"], "outputs": ["constraints"],
    },
    "com.cad.sim.load": {
        "name": "FEA Load",
        "properties": {"load_type": "Force", "force_x": 0.0, "force_y": -1000.0, "force_z": 0.0},
        "inputs": ["mesh", "target_face"], "outputs": ["loads"],
    },
    "com.cad.sim.pressure_load": {
        "name": "FEA Pressure Load",
        "properties": {"pressure": 1000000.0},
        "inputs": ["mesh", "target_face"], "outputs": ["loads"],
    },
    "com.cad.sim.solver": {
        "name": "FEA Solver",
        "properties": {"visualization": "Von Mises Stress"},
        "inputs": ["mesh", "material", "constraints", "loads"], "outputs": ["results"],
    },
    "com.cad.sim.topopt": {
        "name": "Topology Optimization",
        "properties": {"vol_frac": 0.4, "iterations": 50, "filter_radius": 3.0,
                        "density_cutoff": 0.3, "shape_recovery": True},
        "inputs": ["mesh", "material", "constraints", "loads"],
        "outputs": ["optimized_mesh", "recovered_shape"],
    },
    "com.cad.sim.remesh": {
        "name": "Remesh Surface",
        "properties": {"element_size": 3.0, "mesh_quality": "Medium"},
        "inputs": ["topopt_result"], "outputs": ["mesh", "shape"],
    },

    # --- IO / Values ---
    "com.cad.number": {
        "name": "Number",
        "properties": {"value": 10.0},
        "outputs": ["value"],
    },
    "com.cad.variable": {
        "name": "Variable",
        "properties": {"value": 0.0},
        "outputs": ["value"],
    },
    "com.cad.export_step": {
        "name": "Export STEP",
        "properties": {"filename": "output.step"},
        "inputs": ["shape"],
    },
    "com.cad.export_stl": {
        "name": "Export STL",
        "properties": {"filename": "output.stl", "smoothing": 10},
        "inputs": ["shape"],
    },
    "com.cad.import_step": {
        "name": "Import STEP",
        "properties": {"filepath": ""},
        "outputs": ["shape_out"],
    },
    "com.cad.import_stl": {
        "name": "Import STL",
        "properties": {"filepath": ""},
        "outputs": ["mesh_out"],
    },

    # --- Advanced CAD ---
    "com.cad.helix": {
        "name": "Helix",
        "properties": {"radius": 10.0, "pitch": 5.0, "helix_height": 50.0},
        "outputs": ["shape"],
    },
    "com.cad.thicken": {
        "name": "Thicken",
        "properties": {"thickness": 2.0},
        "inputs": ["shape_in"], "outputs": ["shape_out"],
    },
    "com.cad.split": {
        "name": "Split Body",
        "properties": {"plane": "XY", "offset": 0.0, "keep": "Both"},
        "inputs": ["shape_in"], "outputs": ["shape_out"],
    },
    "com.cad.pipe": {
        "name": "Pipe",
        "properties": {"outer_radius": 5.0, "inner_radius": 3.0},
        "inputs": ["path_in"], "outputs": ["shape_out"],
    },
    "com.cad.text": {
        "name": "3D Text",
        "properties": {"text_content": "pylcss", "font_size": 10.0, "depth": 2.0},
        "outputs": ["shape_out"],
    },
    "com.cad.math_expression": {
        "name": "Math Expression",
        "properties": {"expression": "x + y"},
        "inputs": ["x", "y", "z"], "outputs": ["result"],
    },
    "com.cad.measure_distance": {
        "name": "Measure Distance",
        "properties": {},
        "inputs": ["shape_a", "shape_b"], "outputs": ["distance"],
    },
    "com.cad.surface_area": {
        "name": "Surface Area",
        "properties": {},
        "inputs": ["shape_in"], "outputs": ["area_out"],
    },
}


# === System Modeling Node Types Reference ===
# Port naming: InputNode/OutputNode/IntermediateNode ports are named after var_name.
# CustomBlockNode ports are in_1, in_2, ... / out_1, out_2, ...
# When connecting: use the var_name as port name for I/O/Intermediate nodes.

SYSTEM_NODE_TYPES = {
    "com.pfd.input": {
        "name": "Design Variable",
        "description": "A design variable (input parameter) with bounds for the design space.",
        "properties": {
            "var_name": {"type": "string", "default": "x", "description": "Variable name (also renames the output port)"},
            "unit": {"type": "string", "default": "-", "description": "Physical unit (pint-compatible, e.g. 'mm', 'kg', 'N/m^2', or '-' for dimensionless)"},
            "min": {"type": "string", "default": "0.0", "description": "Minimum value in design space"},
            "max": {"type": "string", "default": "10.0", "description": "Maximum value in design space"},
        },
        "inputs": [],
        "outputs": ["<var_name>"],  # Port is named after var_name
    },
    "com.pfd.output": {
        "name": "Quantity of Interest (QoI)",
        "description": "An output quantity with optional requirement bounds and optimization objective.",
        "properties": {
            "var_name": {"type": "string", "default": "y", "description": "Variable name (also renames the input port)"},
            "unit": {"type": "string", "default": "-", "description": "Physical unit (pint-compatible)"},
            "req_min": {"type": "string", "default": "-1e9", "description": "Requirement lower bound (use -1e9 for unconstrained)"},
            "req_max": {"type": "string", "default": "1e9", "description": "Requirement upper bound (use 1e9 for unconstrained)"},
            "minimize": {"type": "boolean", "default": False, "description": "Set True to minimize this QoI (objective)"},
            "maximize": {"type": "boolean", "default": False, "description": "Set True to maximize this QoI (objective)"},
        },
        "inputs": ["<var_name>"],  # Port is named after var_name
        "outputs": [],
    },
    "com.pfd.intermediate": {
        "name": "Intermediate Variable",
        "description": "A pass-through variable connecting functions. Used for chaining black-box outputs to inputs.",
        "properties": {
            "var_name": {"type": "string", "default": "z", "description": "Variable name (renames both ports)"},
            "unit": {"type": "string", "default": "-", "description": "Physical unit (pint-compatible)"},
        },
        "inputs": ["<var_name>"],
        "outputs": ["<var_name>"],
    },
    "com.pfd.custom_block": {
        "name": "Black Box Function",
        "description": "A Python function block with configurable inputs/outputs. Write code using in_1, in_2, ... as inputs and assign to out_1, out_2, ... as outputs.",
        "properties": {
            "num_inputs": {"type": "string", "default": "1", "description": "Number of input ports (creates in_1, in_2, ...)"},
            "num_outputs": {"type": "string", "default": "1", "description": "Number of output ports (creates out_1, out_2, ...)"},
            "code_content": {"type": "string", "default": "# out_1 = in_1 * 2\n",
                             "description": "Python code. Use in_1, in_2,... as input variables. Assign results to out_1, out_2,... "
                                            "Supports numpy (np), math. Example: 'out_1 = in_1**2 + in_2'"},
            "use_surrogate": {"type": "boolean", "default": False, "description": "Use trained surrogate model instead of code"},
        },
        "inputs": ["in_1", "in_2", "..."],   # Dynamic: in_1 to in_N
        "outputs": ["out_1", "out_2", "..."],  # Dynamic: out_1 to out_N
    },
}


def get_cad_schema_for_prompt() -> str:
    """Get a compact CAD node schema for LLM prompts.

    Format: one line per node — type (Name) | props: key=default | in→out
    The `com.cad.` prefix is stripped; it will be auto-added back on input.
    """
    lines = ["## CAD Nodes  (type | props | ports)"]
    lines.append("Note: all types are prefixed `com.cad.` internally — you write them WITHOUT the prefix.")
    lines.append("")

    # Core nodes that the LLM should know in detail
    CORE_NODES = [
        # Primitives
        "com.cad.box", "com.cad.cylinder", "com.cad.sphere", "com.cad.cone",
        "com.cad.torus", "com.cad.wedge", "com.cad.pyramid",
        # Sketch
        "com.cad.sketch", "com.cad.sketch.circle", "com.cad.sketch.rectangle",
        "com.cad.sketch.polygon", "com.cad.polyline",
        # 3D ops
        "com.cad.extrude", "com.cad.revolve", "com.cad.boolean",
        "com.cad.fillet", "com.cad.chamfer", "com.cad.shell",
        "com.cad.sweep", "com.cad.loft", "com.cad.select_face",
        # Cutting
        "com.cad.hole_at_coords", "com.cad.multi_hole", "com.cad.array_holes",
        "com.cad.cylinder_cut",
        # Transforms
        "com.cad.translate", "com.cad.rotate", "com.cad.scale", "com.cad.mirror",
        # Patterns
        "com.cad.linear_pattern", "com.cad.circular_pattern",
        # Values
        "com.cad.number", "com.cad.variable",
        # IO
        "com.cad.export_step", "com.cad.export_stl",
        # FEA
        "com.cad.sim.material", "com.cad.sim.mesh", "com.cad.sim.constraint",
        "com.cad.sim.load", "com.cad.sim.solver", "com.cad.sim.topopt",
    ]

    for ntype in CORE_NODES:
        info = CAD_NODE_TYPES.get(ntype)
        if not info:
            continue
        props = info.get("properties", {})
        ins = info.get("inputs", [])
        outs = info.get("outputs", ["shape"])

        # Strip com.cad. prefix for display
        short = ntype.replace("com.cad.", "", 1)

        # Compact prop string: key=val
        pstr = ", ".join(f"{k}={v}" for k, v in props.items())
        istr = ",".join(ins) if ins else "—"
        ostr = ",".join(outs)
        lines.append(f"- `{short}` ({info.get('name','')}) | {pstr} | {istr}→{ostr}")

    # List remaining node types in a brief "Also available" block
    extra = [nt.replace("com.cad.", "", 1) for nt in CAD_NODE_TYPES if nt not in CORE_NODES]
    if extra:
        lines.append(f"\nAlso available: {', '.join(extra)}")

    return "\n".join(lines)


def get_modeling_schema_for_prompt() -> str:
    """Get system modeling node schema as text for LLM prompts."""
    lines = [
        "## System Modeling Node Types",
        "",
        "### Port Naming Convention",
        "- **InputNode** output port is named after `var_name` (e.g. var_name='width' → port 'width')",
        "- **OutputNode** input port is named after `var_name` (e.g. var_name='mass' → port 'mass')",
        "- **IntermediateNode** both ports are named after `var_name`",
        "- **CustomBlockNode** ports are `in_1, in_2, ...` and `out_1, out_2, ...`",
        "",
        "### Connection Format",
        "  `{\"from\": \"<node_id>.<port_name>\", \"to\": \"<node_id>.<port_name>\"}`",
        "  Example: `{\"from\": \"x.x\", \"to\": \"fn.in_1\"}` connects InputNode 'x' to CustomBlock 'fn'",
        "",
    ]

    for ntype, info in SYSTEM_NODE_TYPES.items():
        lines.append(f"### {ntype} — {info['name']}")
        lines.append(f"  {info['description']}")
        lines.append("  Properties:")
        for pname, pinfo in info["properties"].items():
            lines.append(f"    - `{pname}` ({pinfo['type']}, default={pinfo['default']}): {pinfo['description']}")
        if info["inputs"]:
            lines.append(f"  Inputs: {', '.join(info['inputs'])}")
        if info["outputs"]:
            lines.append(f"  Outputs: {', '.join(info['outputs'])}")
        lines.append("")

    return "\n".join(lines)
