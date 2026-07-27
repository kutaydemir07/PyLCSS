# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Graph execution engine with dirty-state caching and simulation control."""
from collections import deque
import hashlib
import pickle


class GraphExecutionCancelled(RuntimeError):
    """Raised when a graph worker stops between safe node boundaries."""

    def __init__(self, results=None):
        super().__init__("Graph execution cancelled by the user.")
        self.results = dict(results or {})

# Node identifiers for simulation nodes (skip during auto-update)
SIMULATION_NODE_IDENTIFIERS = {
    'com.cad.sim.material',
    'com.cad.sim.mesh',
    'com.cad.sim.remesh',
    'com.cad.sim.constraint',
    'com.cad.sim.load',
    'com.cad.sim.pressure_load',
    'com.cad.sim.solver',
    'com.cad.sim.topopt_voxel',
    # Crash / Impact nodes
    'com.cad.sim.crash_material',
    'com.cad.sim.impact',
    'com.cad.sim.crash_solver',
    # Standalone decks can launch Starter + Engine without any graph inputs,
    # so omitting this identifier makes project-open/auto-preview execute them.
    'com.cad.sim.radioss_deck',
}

def _hash_value(value):
    """Create a hash of a value for change detection."""
    try:
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(data).hexdigest()
    except Exception:
        return hashlib.md5(repr(value).encode("utf-8", errors="replace")).hexdigest()


_NON_EXECUTION_PROPERTIES = {
    # Viewer/post-processing settings update cached results directly and must
    # not invalidate an expensive engineering solve.
    'visualization', 'deformation_scale', 'disp_scale', 'density_cutoff',
    'advanced_settings_visible', 'cad_export_filename',
    'cad_reconstruction_method',
    # Hidden compatibility-only properties retained for old .cad sessions.
    'solver_backend', 'moment_x', 'moment_y', 'moment_z',
    'damping_alpha', 'damping_beta', 'enable_corotation', 'enable_contact',
    'contact_stiffness', 'contact_thickness', 'contact_update_interval',
    'mass_scaling_threshold', 'bc_preset', 'quality_preset',
    # Internal UI/error bookkeeping.
    'error_state', 'error_message',
}


def _execution_properties(node):
    """Return only custom properties that can affect ``node.run()``."""
    try:
        props = node.properties()
        custom = dict(props.get('custom', {}) or {})
    except Exception:
        return {}
    return {
        key: value for key, value in custom.items()
        if key not in _NON_EXECUTION_PROPERTIES
    }

def _is_simulation_node(node):
    """Check if a node is a simulation node that should be skipped during auto-update."""
    identifier = getattr(node, '__identifier__', '')
    return identifier in SIMULATION_NODE_IDENTIFIERS


def _connected_upstream_nodes(node):
    """Yield nodes connected to any input port."""
    if not hasattr(node, 'input_ports'):
        return
    inputs = node.input_ports()
    if isinstance(inputs, dict):
        inputs = list(inputs.values())
    for inp in inputs:
        if not hasattr(inp, 'connected_ports'):
            continue
        for cp in inp.connected_ports():
            try:
                yield cp.node()
            except Exception:
                continue


# Light-weight nodes that should still run during preview/skip-simulation
# updates even when their upstream is a simulation node (e.g. Remesh).  Without
# this carve-out, picking faces on an STL → Remesh → InteractiveSelectFace
# pipeline never refreshes the picker's _last_result after the user clicks
# Done, and the BC overlay shows the previous (or empty) selection.
#
# Mesh / Remesh are preview-safe so that selecting a Mesh node (or any node
# while a Mesh node is present) computes and shows the mesh in the viewer
# instead of falling back to the upstream CAD solid.  They remain "simulation
# nodes" for every other purpose; the hash-based cache in execute_graph means
# they only re-mesh when the shape or element size actually changes, so the
# fast CAD preview is not re-meshed on unrelated edits.
PREVIEW_SAFE_IDENTIFIERS = {
    'com.cad.select_face',
    'com.cad.select_face_interactive',
    'com.cad.sim.mesh',
    'com.cad.sim.remesh',
}


def _is_preview_safe(node):
    return getattr(node, '__identifier__', '') in PREVIEW_SAFE_IDENTIFIERS


def _filter_for_preview(nodes):
    """Skip simulation nodes and all downstream consumers during preview.

    Previously preview mode removed only the simulation nodes themselves.  That
    left Export STEP / STL and other downstream nodes in the execution list;
    their input resolvers could then call heavy upstream TopOpt/FEA nodes
    directly, bypassing the worker's progress callback and confusing the GUI.

    Face-selector nodes are exempt from the downstream-of-simulation rule:
    they only read cached mesh patches and are essential for refreshing the
    picker's _last_result after the user picks faces on a remeshed surface.

    Preview-safe simulation nodes (Mesh / Remesh) are also exempt from the
    initial block so the mesh itself is generated and rendered during preview;
    their heavy downstream consumers (Solver, Constraint, Load, …) stay blocked
    because those are simulation nodes that are not preview-safe.
    """
    blocked = {n for n in nodes
               if _is_simulation_node(n) and not _is_preview_safe(n)}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node in blocked:
                continue
            if _is_preview_safe(node):
                continue
            if any(upstream in blocked for upstream in _connected_upstream_nodes(node)):
                blocked.add(node)
                changed = True
    return [n for n in nodes if n not in blocked]


def _invalidate_downstream_cache(node, reverse_dependencies):
    """Clear cached results for nodes affected by an upstream execution failure."""
    pending = deque(reverse_dependencies.get(node, ()))
    visited = set()

    while pending:
        downstream = pending.popleft()
        if downstream in visited:
            continue
        visited.add(downstream)

        setattr(downstream, '_last_result', None)
        setattr(downstream, '_last_input_hash', None)
        setattr(downstream, '_dirty', True)
        setattr(downstream, '_force_execute', False)

        if hasattr(downstream, 'clear_error'):
            try:
                downstream.clear_error()
            except Exception:
                pass

        pending.extend(reverse_dependencies.get(downstream, ()))

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
        nodes = list(graph_or_nodes)

    # Filter out simulation nodes and their downstream consumers if
    # skip_simulation is True.
    if skip_simulation:
        nodes_to_execute = _filter_for_preview(nodes)
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

    dependencies = {node: set(upstream) for node, upstream in deps.items()}

    # 2. Topological Sort (Kahn's Algorithm)
    remaining_deps = {node: set(upstream) for node, upstream in deps.items()}
    ready = deque([n for n, d in remaining_deps.items() if not d])
    order = []

    while ready:
        n = ready.popleft()
        order.append(n)
        for m in list(rev.get(n, [])):
            remaining_deps[m].discard(n)
            if not remaining_deps[m]:
                ready.append(m)

    # A cyclic engineering graph has no valid execution order.  Running it in
    # insertion order made connected input resolvers recurse unpredictably and
    # could launch solvers with stale inputs.
    if len(order) != len(nodes_to_execute):
        cyclic = [n for n in nodes_to_execute if remaining_deps.get(n)]
        names = []
        for node in cyclic[:10]:
            try:
                names.append(node.name() if callable(node.name) else str(node.name))
            except Exception:
                names.append(node.__class__.__name__)
        suffix = "" if len(cyclic) <= 10 else f" and {len(cyclic) - 10} more"
        raise RuntimeError(
            "Graph contains a dependency cycle involving: "
            + ", ".join(names)
            + suffix
            + ". Remove the feedback connection before running."
        )

    # 3. Execution with Deep Hash-Based Caching
    cancel_callback = kwargs.pop('cancel_callback', None)
    kwargs.setdefault('preview', bool(skip_simulation))
    results = {}
    failed_nodes = set()
    errors = []

    for n in order:
        if callable(cancel_callback) and cancel_callback():
            raise GraphExecutionCancelled(results)

        failed_upstream = [up for up in dependencies.get(n, ()) if up in failed_nodes]
        if failed_upstream:
            upstream_names = []
            for upstream in failed_upstream:
                try:
                    upstream_names.append(
                        upstream.name() if callable(upstream.name) else str(upstream.name)
                    )
                except Exception:
                    upstream_names.append(upstream.__class__.__name__)
            message = "Skipped because upstream failed: " + ", ".join(upstream_names)
            setattr(n, '_last_result', None)
            setattr(n, '_last_input_hash', None)
            setattr(n, '_dirty', True)
            setattr(n, '_force_execute', False)
            if hasattr(n, 'set_error'):
                n.set_error(message)
            failed_nodes.add(n)
            continue

        # Collect current input values for hashing
        input_signature = []
        if hasattr(n, 'input_ports'):
            inputs = n.input_ports()
            if isinstance(inputs, dict):
                inputs = list(inputs.values())

            for inp in inputs:
                if hasattr(inp, 'connected_ports'):
                    for cp in inp.connected_ports():
                        upstream_node = cp.node()
                        input_signature.append((
                            id(upstream_node),
                            cp.name(),
                            int(getattr(upstream_node, '_result_revision', 0)),
                        ))

        input_signature.sort(key=lambda value: (value[0], value[1]))
        current_input_hash = _hash_value((
            input_signature,
            _execution_properties(n),
            bool(skip_simulation),
        ))

        # Check if node needs execution
        last_input_hash = getattr(n, '_last_input_hash', None)
        has_cached_result = hasattr(n, '_last_result')
        cached_result = getattr(n, '_last_result', None)

        can_skip = (current_input_hash == last_input_hash and
                   has_cached_result and
                   not getattr(n, '_dirty', False) and
                   not getattr(n, '_force_execute', False))

        if can_skip:
            results[n] = cached_result
            continue

        # Execute the node
        try:
            # Clear only stale errors *before* running.  Several nodes report
            # validation/backend failures by calling set_error() and returning
            # None instead of raising.  Clearing after run used to erase those
            # fresh errors and made the GUI report "Computation complete".
            if hasattr(n, 'clear_error'):
                n.clear_error()

            # Check if run() accepts kwargs (e.g., progress_callback)
            import inspect
            sig = inspect.signature(n.run)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            
            if valid_kwargs:
                res = n.run(**valid_kwargs)
            else:
                res = n.run()

            node_has_error = False
            if hasattr(n, 'has_error'):
                try:
                    node_has_error = bool(n.has_error())
                except Exception:
                    node_has_error = False
            if node_has_error:
                message = None
                if hasattr(n, 'get_error'):
                    try:
                        message = n.get_error()
                    except Exception:
                        message = None

            setattr(n, '_last_result', res)
            setattr(n, '_last_input_hash', current_input_hash)
            setattr(n, '_result_revision', int(getattr(n, '_result_revision', 0)) + 1)
            setattr(n, '_dirty', False)
            setattr(n, '_force_execute', False)
            results[n] = res
        except Exception as e:
            setattr(n, '_last_result', None)
            setattr(n, '_last_input_hash', None)
            setattr(n, '_dirty', True)
            setattr(n, '_force_execute', False)
            _invalidate_downstream_cache(n, rev)
            if hasattr(n, 'set_error'):
                n.set_error(str(e))
            node_name = getattr(n, 'name', None)
            if callable(node_name):
                node_name = node_name()
            if not node_name:
                node_name = getattr(n, 'NODE_NAME', None) or n.__class__.__name__
            errors.append((str(node_name), str(e)))
            failed_nodes.add(n)

    if callable(cancel_callback) and cancel_callback():
        raise GraphExecutionCancelled(results)

    if errors:
        pass

    return results
