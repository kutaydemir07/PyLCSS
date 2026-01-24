import logging
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .base_tool import BaseTool
from pylcss.cad.node_library import NODE_CLASS_MAPPING
from pylcss.system_modeling.node_registry import SYSTEM_NODE_CLASS_MAPPING

logger = logging.getLogger(__name__)

class NodeSpec(BaseModel):
    id: str = Field(..., description="Unique ID for the node")
    type: str = Field(..., description="Type of the node (e.g., com.cad.box)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Properties to set on the node")

class ConnectionSpec(BaseModel):
    from_node: str = Field(..., description="ID of source node")
    from_port: str = Field(default="shape", description="Source port name")
    to_node: str = Field(..., description="ID of target node")
    to_port: str = Field(default="shape", description="Target port name")

class CreateGraphSchema(BaseModel):
    environment: str = Field(..., description="Target environment: 'cad' or 'modeling'")
    nodes: List[NodeSpec] = Field(..., description="List of nodes to create")
    connections: List[ConnectionSpec] = Field(default_factory=list, description="List of connections to make")

class CreateGraphTool(BaseTool):
    name = "create_graph"
    description = "Create or update a graph of nodes in the CAD or Modeling environment."
    args_schema = CreateGraphSchema

    def __init__(self, main_window=None):
        self.main_window = main_window

    def run(self, environment: str, nodes: List[NodeSpec], connections: List[ConnectionSpec]) -> str:
        if not self.main_window:
            return "Error: Main window not available."

        # Route to the correct internal method based on environment
        # We reuse the logic from CommandDispatcher but encapsulated here
        # Ideally, we should move the logic entirely here, but for now we can call back 
        # or implement it here using the main_window reference.
        
        # Accessing widgets safely
        if environment.lower() == 'cad':
            return self._build_cad_graph(nodes, connections)
        elif environment.lower() == 'modeling':
            return self._build_modeling_graph(nodes, connections)
        else:
            return f"Error: Unknown environment '{environment}'"

    def _get_start_pos(self, graph) -> tuple[int, int]:
        existing = graph.all_nodes()
        if existing:
            return 0, max((n.pos()[1] for n in existing), default=0) + 200
        return 0, 0

    def _build_cad_graph(self, nodes: List[NodeSpec], connections: List[ConnectionSpec]) -> str:
        if not hasattr(self.main_window, 'cad_widget'):
            return "Error: CAD widget not found."
            
        cad_widget = self.main_window.cad_widget
        graph = getattr(cad_widget, 'graph', None)
        if not graph:
            return "Error: CAD graph controller not found."

        # Switch tab (optional but good for UX)
        # We assume this runs on the GUI thread or via invokeMethod
        # For this tool implementation, we will assume it's called safely or we need to dispatch
        
        # IMPORTANT: Tools should probably return a result or command object, 
        # but if we execute directly we need QMetaObject.invokeMethod if we are in a background thread.
        # However, the CommandDispatcher called the tool, and CommandDispatcher methods 
        # often use invokeMethod or run in main thread. 
        # Let's assume the tool is responsible for thread safety if it touches GUI.

        from PySide6.QtCore import QMetaObject, Qt
        
        # We'll use a helper to run on main thread if needed
        # For simplicity, let's implement the logic assuming we are on the main thread 
        # or we wrap the whole thing.
        # Since the CommandDispatcher currently does heavy lifting, let's try to REUSE 
        # the CommandDispatcher's _build_node_graph if possible, OR re-implement it cleaner here.
        # Re-implementing clearer is better for the long term.

        created_info = []

        # Map IDs to actual nodes
        id_to_node = {n.name(): n for n in graph.all_nodes()} # Start with existing
        
        start_x, start_y = self._get_start_pos(graph)
        count = 0

        for spec in nodes:
            node = None
            if spec.id in id_to_node:
                node = id_to_node[spec.id]
                # Update properties
                for k, v in spec.properties.items():
                   if hasattr(node, "set_property"):
                       node.set_property(k, v)
                created_info.append(f"Updated {spec.id}")
            else:
                # Create
                node_class = NODE_CLASS_MAPPING.get(spec.type)
                if not node_class:
                    logger.warning(f"Unknown CAD node type: {spec.type}")
                    continue
                
                node = node_class()
                row = count // 4
                col = count % 4
                node.set_pos(start_x + (col * 250), start_y + (row * 150))
                graph.add_node(node)
                
                # Try to set name
                node.set_name(spec.id)
                count += 1
                
                id_to_node[spec.id] = node
                
                # Set properties with robust logic
                for k, v in spec.properties.items():
                    if not hasattr(node, "has_property"):
                         continue
                         
                    # 1. Normalize Property Name (Case-insensitive)
                    final_prop_name = k
                    if not node.has_property(k):
                        # Try to find case-insensitive match
                        for p in node.properties().keys():
                            if p.lower() == k.lower():
                                final_prop_name = p
                                break
                    
                    if not node.has_property(final_prop_name):
                         logger.warning(f"Property '{k}' not found on {node.name()}")
                         continue

                    # 2. Normalize Value (Synonyms & Case-insensitivity)
                    final_val = v
                    
                    # Enum Mapping for common properties
                    ENUM_MAP = {
                        "operation": {
                            "targets": ["Union", "Cut", "Intersect"],
                            "synonyms": {
                                "difference": "Cut", "subtract": "Cut", "subtraction": "Cut", "remove": "Cut",
                                "add": "Union", "addition": "Union", "combine": "Union",
                                "intersection": "Intersect"
                            }
                        },
                        "selector_type": {
                            "targets": ["Direction", "NearestToPoint", "Index", "Largest Area", "Tag"],
                            "synonyms": {}
                        },
                    }
                    
                    # Check if this property has an enum map
                    prop_key_lower = final_prop_name.lower()
                    map_entry = None
                    for known_key, config in ENUM_MAP.items():
                        if known_key in prop_key_lower: # partial match ok (e.g. "boolean_operation")
                             map_entry = config
                             break
                    
                    if map_entry and isinstance(v, str):
                        val_lower = v.lower()
                        # Check direct targets
                        found = False
                        for target in map_entry["targets"]:
                            if target.lower() == val_lower:
                                final_val = target
                                found = True
                                break
                        # Check synonyms
                        if not found and val_lower in map_entry["synonyms"]:
                            final_val = map_entry["synonyms"][val_lower]

                    # 3. Apply
                    try:
                        node.set_property(final_prop_name, final_val)
                    except Exception as e:
                        logger.error(f"Failed to set {final_prop_name} on {node.name()}: {e}")

                created_info.append(f"Created {spec.id}")

        # Connections
        for conn in connections:
            src = id_to_node.get(conn.from_node)
            dst = id_to_node.get(conn.to_node)
            
            if src and dst:
                out = src.get_output(conn.from_port)
                # Fallbacks for common port names
                if not out and conn.from_port == 'shape':
                     out = src.get_output('result') or src.get_output('out')
                
                inp = dst.get_input(conn.to_port)
                if not inp and conn.to_port == 'shape':
                     inp = dst.get_input('input_shape') or dst.get_input('target')
                
                if out and inp:
                    out.connect_to(inp)
                    
        # Apply Auto-Layout for new nodes
        # Filter to only the nodes we just created/touched to avoid messing up user's existing layout too much
        # or just layout everything? Better just the batch.
        batch_nodes = [id_to_node[spec.id] for spec in nodes if spec.id in id_to_node]
        self._apply_layout(batch_nodes, connections, start_x, start_y)
        
        return f"Graph operation complete: {', '.join(created_info)}"

    def _apply_layout(self, nodes: List, connections: List[ConnectionSpec], start_x: int, start_y: int):
        """
        Simple layered layout algorithm (Left-to-Right).
        """
        if not nodes:
            return

        # Build local adjacency
        node_ids = {n.name(): n for n in nodes} # Assumes name matches ID
        adj = {n.name(): [] for n in nodes}
        in_degree = {n.name(): 0 for n in nodes}
        
        for conn in connections:
            u, v = conn.from_node, conn.to_node
            if u in adj and v in in_degree:
                adj[u].append(v)
                in_degree[v] += 1
        
        # Assign ranks & topological sort roughly
        queue = [n_id for n_id, d in in_degree.items() if d == 0]
        ranks = {n_id: 0 for n_id in node_ids}
        
        processed = set()
        while queue:
            curr = queue.pop(0)
            processed.add(curr)
            
            curr_rank = ranks[curr]
            
            for neighbor in adj[curr]:
                ranks[neighbor] = max(ranks[neighbor], curr_rank + 1)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Handle cycles/remaining nodes (just default them to next rank)
        max_rank = 0
        if ranks:
             max_rank = max(ranks.values())
             
        for n_id in node_ids:
            if n_id not in processed:
                 ranks[n_id] = max_rank + 1

        # Group by rank
        rank_groups = {}
        for n_id, r in ranks.items():
            if r not in rank_groups: rank_groups[r] = []
            rank_groups[r].append(n_id)
            
        # Assign Positions
        X_SPACING = 350
        Y_SPACING = 200
        
        for r in sorted(rank_groups.keys()):
            group = rank_groups[r]
            # Center group vertically around start_y? Or just stack down
            # Let's stack down
            for i, n_id in enumerate(group):
                x = start_x + (r * X_SPACING)
                y = start_y + (i * Y_SPACING)
                if n_id in node_ids:
                    node_ids[n_id].set_pos(x, y)

    def _build_modeling_graph(self, nodes: List[NodeSpec], connections: List[ConnectionSpec]) -> str:
        if not hasattr(self.main_window, 'modeling_widget'):
            return "Error: Modeling widget not found."
            
        widget = self.main_window.modeling_widget
        graph = getattr(widget, 'current_graph', None)
        if not graph:
            return "Error: Modeling graph controller not found."

        created_info = []
        id_to_node = {n.name(): n for n in graph.all_nodes()}
        
        start_x, start_y = self._get_start_pos(graph)
        count = 0

        for spec in nodes:
            if spec.id in id_to_node:
                node = id_to_node[spec.id]
                for k, v in spec.properties.items():
                    if hasattr(node, "set_property"):
                        node.set_property(k, v)
                created_info.append(f"Updated {spec.id}")
            else:
                node_class = SYSTEM_NODE_CLASS_MAPPING.get(spec.type)
                if not node_class:
                    logger.warning(f"Unknown System node type: {spec.type}")
                    continue
                
                node = node_class()
                row = count // 4
                col = count % 4
                node.set_pos(start_x + (col * 250), start_y + (row * 150))
                graph.add_node(node)
                node.set_name(spec.id)
                count += 1
                
                id_to_node[spec.id] = node
                
                for k, v in spec.properties.items():
                     if node.has_property(k):
                        node.set_property(k, v)
                
                created_info.append(f"Created {spec.id}")
        
        # Connections logic similar to CAD but simpler ports usually
        for conn in connections:
            src = id_to_node.get(conn.from_node)
            dst = id_to_node.get(conn.to_node)
            if src and dst:
                out = src.get_output(conn.from_port)
                inp = dst.get_input(conn.to_port)
                if out and inp:
                    out.connect_to(inp)

        # Apply Layout
        batch_nodes = [id_to_node[spec.id] for spec in nodes if spec.id in id_to_node]
        self._apply_layout(batch_nodes, connections, start_x, start_y)

        return f"Modeling Graph operation complete: {', '.join(created_info)}"
