# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Graph execution engine with dirty-state caching and simulation control."""
from collections import deque
import hashlib
import pickle

# Node identifiers for simulation nodes (skip during auto-update)
SIMULATION_NODE_IDENTIFIERS = {
    'com.cad.sim.material',
    'com.cad.sim.mesh',
    'com.cad.sim.constraint',
    'com.cad.sim.load',
    'com.cad.sim.pressure_load',
    'com.cad.sim.solver',
    'com.cad.sim.topopt',
}

def _hash_value(value):
    """Create a hash of a value for change detection."""
    try:
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(data).hexdigest()
    except (TypeError, pickle.PicklingError):
        return hashlib.md5(str(value).encode()).hexdigest()

def _is_simulation_node(node):
    """Check if a node is a simulation node that should be skipped during auto-update."""
    identifier = getattr(node, '__identifier__', '')
    return identifier in SIMULATION_NODE_IDENTIFIERS

def execute_graph(graph_or_nodes, skip_simulation=False, **kwargs):
    """
    Execute nodes in topological order with deep hash-based dirty checks.
    
    Args:
        graph_or_nodes: NodeGraph object or list of nodes
        skip_simulation: If True, skip FEA/TopOpt nodes (for auto-update mode)
        **kwargs: Additional arguments passed to node.run() if supported (e.g. progress_callback)
    
    Returns:
        dict: Results from executed nodes
    """
    # Handle both Graph object and List of nodes
    if hasattr(graph_or_nodes, 'all_nodes'):
        nodes = list(graph_or_nodes.all_nodes())
    else:
        nodes = graph_or_nodes

    # Filter out simulation nodes if skip_simulation is True
    if skip_simulation:
        nodes_to_execute = [n for n in nodes if not _is_simulation_node(n)]
    else:
        nodes_to_execute = nodes

    deps = {n: set() for n in nodes_to_execute}
    rev = {n: set() for n in nodes_to_execute}

    for n in nodes_to_execute:
        if not hasattr(n, 'input_ports'):
            continue

        inputs = n.input_ports()
        if isinstance(inputs, dict):
            inputs = list(inputs.values())

        for inp in inputs:
            if not hasattr(inp, 'connected_ports'):
                continue

            for cp in inp.connected_ports():
                dep_node = cp.node()
                # Only add dependency if it's in our execution list
                if dep_node in deps:
                    deps[n].add(dep_node)
                    rev[dep_node].add(n)

    # 2. Topological Sort (Kahn's Algorithm)
    ready = deque([n for n, d in deps.items() if not d])
    order = []

    while ready:
        n = ready.popleft()
        order.append(n)
        for m in list(rev.get(n, [])):
            deps[m].discard(n)
            if not deps[m]:
                ready.append(m)

    # Handle cycles
    if len(order) != len(nodes_to_execute):
        order = nodes_to_execute

    # 3. Execution with Deep Hash-Based Caching
    results = {}
    executed_nodes = set()

    for n in order:
        # Collect current input values for hashing
        current_input_hash = ""
        if hasattr(n, 'input_ports'):
            inputs = n.input_ports()
            if isinstance(inputs, dict):
                inputs = list(inputs.values())

            input_values = []
            for inp in inputs:
                if hasattr(inp, 'connected_ports'):
                    for cp in inp.connected_ports():
                        upstream_node = cp.node()
                        upstream_result = getattr(upstream_node, '_last_result', None)
                        if upstream_result is not None:
                            input_values.append((cp.name(), upstream_result))

            input_values.sort(key=lambda x: x[0])
            current_input_hash = _hash_value(input_values)

        # Check if node needs execution
        last_input_hash = getattr(n, '_last_input_hash', None)
        cached_result = getattr(n, '_last_result', None)

        can_skip = (current_input_hash == last_input_hash and
                   cached_result is not None and
                   not getattr(n, '_force_execute', False))

        if can_skip:
            results[n] = cached_result
            continue

        # Execute the node
        try:
            # Check if run() accepts kwargs (e.g., progress_callback)
            import inspect
            sig = inspect.signature(n.run)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            
            if valid_kwargs:
                res = n.run(**valid_kwargs)
            else:
                res = n.run()

            setattr(n, '_last_result', res)
            setattr(n, '_last_input_hash', current_input_hash)
            setattr(n, '_dirty', False)
            setattr(n, '_force_execute', False)
            executed_nodes.add(n)

            if hasattr(n, 'clear_error'):
                n.clear_error()
                
            results[n] = res
        except Exception as e:
            if hasattr(n, 'set_error'):
                n.set_error(str(e))
            
    return results