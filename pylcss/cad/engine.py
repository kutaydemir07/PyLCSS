# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Graph execution engine with dirty-state caching."""
from collections import deque
import hashlib
import pickle

def _hash_value(value):
    """Create a hash of a value for change detection."""
    try:
        # Use pickle to serialize the value, then hash it
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(data).hexdigest()
    except (TypeError, pickle.PicklingError):
        # If pickling fails, use string representation
        return hashlib.md5(str(value).encode()).hexdigest()

def execute_graph(graph_or_nodes):
    """
    Execute nodes in topological order with deep hash-based dirty checks.
    Accepts either a NodeGraph object or a list of nodes.
    Only re-executes nodes when their inputs actually change.
    """
    # CHANGE: Handle both Graph object and List of nodes
    if hasattr(graph_or_nodes, 'all_nodes'):
        nodes = list(graph_or_nodes.all_nodes())
    else:
        nodes = graph_or_nodes # It's already a list from our worker
    deps = {n: set() for n in nodes}
    rev = {n: set() for n in nodes}

    for n in nodes:
        # Check if node has inputs
        if not hasattr(n, 'input_ports'):
            continue

        # Handle input_ports returning list or dict
        inputs = n.input_ports()
        if isinstance(inputs, dict):
            inputs = list(inputs.values())

        for inp in inputs:
            if not hasattr(inp, 'connected_ports'):
                continue

            for cp in inp.connected_ports():
                dep_node = cp.node()
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
    if len(order) != len(nodes):
        order = nodes

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

            # Sort for consistent hashing
            input_values.sort(key=lambda x: x[0])
            current_input_hash = _hash_value(input_values)

        # Check if node needs execution
        last_input_hash = getattr(n, '_last_input_hash', None)
        cached_result = getattr(n, '_last_result', None)

        # Only skip if inputs haven't changed AND we have a cached result
        can_skip = (current_input_hash == last_input_hash and
                   cached_result is not None and
                   not getattr(n, '_force_execute', False))

        if can_skip:
            results[n] = cached_result
            continue

        # Execute the node
        try:
            res = n.run()

            # Update cache with new result and input hash
            setattr(n, '_last_result', res)
            setattr(n, '_last_input_hash', current_input_hash)
            setattr(n, '_dirty', False)
            setattr(n, '_force_execute', False)
            executed_nodes.add(n)

            # Clear error state
            if hasattr(n, 'clear_error'):
                n.clear_error()
                
            results[n] = res
        except Exception as e:
            # Set error state on the node
            if hasattr(n, 'set_error'):
                n.set_error(str(e))
            
    return results