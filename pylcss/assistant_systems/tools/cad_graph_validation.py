# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Normalize, repair, and validate assistant-authored Design Studio graphs."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from pylcss.assistant_systems.tools.cad_node_catalog import CAD_NODE_TYPES
from pylcss.assistant_systems.tools.graph_normalization import normalize_node_specs

if TYPE_CHECKING:
    from pylcss.assistant_systems.api.dispatcher import CommandDispatcher

logger = logging.getLogger(__name__)


def _compact_graph_view(data: dict[str, Any]) -> dict[str, Any]:
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        nodes = []
    connections = data.get("connections", [])
    if not isinstance(connections, list):
        connections = []
    compact_nodes = []
    for node in nodes[:12]:
        if not isinstance(node, dict):
            compact_nodes.append(node)
            continue

        props = node.get("properties", {})
        compact_nodes.append(
            {
                "id": node.get("id"),
                "type": node.get("type"),
                "properties": props,
            }
        )

    view: dict[str, Any] = {
        "node_count": len(nodes),
        "connection_count": len(connections),
        "nodes": compact_nodes,
        "connections": connections[:20],
    }
    if len(nodes) > 12:
        view["nodes_truncated"] = len(nodes) - 12
    if len(connections) > 20:
        view["connections_truncated"] = len(connections) - 20
    return view


def _log_graph_payload(stage: str, data: dict[str, Any], goal: str = "") -> None:
    compact = _compact_graph_view(data)
    message = f"CAD payload {stage}"
    if goal:
        message += f" | goal={goal!r}"
    logger.info(message + f" | summary={compact}")
    try:
        payload_json = json.dumps(compact, ensure_ascii=True, sort_keys=True)
    except TypeError:
        payload_json = str(compact)
    if len(payload_json) > 4000:
        payload_json = payload_json[:4000] + "...<truncated>"
    logger.info(f"CAD payload {stage} json={payload_json}")


def _node_types(data: dict[str, Any]) -> list[str]:
    return [
        str(node.get("type", ""))
        for node in data.get("nodes", [])
        if isinstance(node, dict)
    ]


def _has_goal_term(goal: str, *terms: str) -> bool:
    lowered = goal.lower()
    return any(term in lowered for term in terms)


def _collect_cad_features(data: dict[str, Any]) -> dict[str, bool]:
    types = set(_node_types(data))
    # A com.cad.code_part node can contain any geometry — treat its
    # presence as satisfying all structural feature requirements so
    # semantic rules don't false-positive on code-first graphs.
    has_code_part = "com.cad.code_part" in types
    has_base = bool(
        types
        & {
            "com.cad.geometry.box",
            "com.cad.geometry.cylinder",
            "com.cad.geometry.tube",
            "com.cad.geometry.cylindrical_shell",
        }
    )
    has_rotational = bool(
        types
        & {
            "com.cad.geometry.cylinder",
            "com.cad.geometry.tube",
            "com.cad.geometry.cylindrical_shell",
        }
    )
    return {
        "base_solid": has_code_part or has_base,
        "sketch_profile": has_code_part,
        "additive": has_code_part or has_base,
        "subtractive": has_code_part
        or bool(
            types
            & {
                "com.cad.geometry.through_hole",
                "com.cad.geometry.boolean",
            }
        ),
        "holes": has_code_part or "com.cad.geometry.through_hole" in types,
        "rounded": has_code_part or "com.cad.geometry.fillet" in types,
        "beveled": has_code_part,
        "hollow": has_code_part
        or bool(
            types
            & {
                "com.cad.geometry.tube",
                "com.cad.geometry.cylindrical_shell",
            }
        ),
        "rotational": has_code_part or has_rotational,
        "revolved": has_code_part,
        "swept": has_code_part,
        "lofted": has_code_part,
        "tooth_like": has_code_part,
    }


def _missing_features(
    features: dict[str, bool],
    required_any: list[str] | None = None,
    required_all: list[str] | None = None,
) -> bool:
    if required_all and any(not features.get(name, False) for name in required_all):
        return True
    if required_any and not any(features.get(name, False) for name in required_any):
        return True
    return False


def _verify_cad_semantics(data: dict[str, Any], goal: str = "") -> list[str]:
    """Check whether the CAD graph matches key semantic intent from the goal."""
    if not goal:
        return []

    issues: list[str] = []
    lowered_goal = goal.lower()

    features = _collect_cad_features(data)
    semantic_rules = [
        {
            "terms": ["gear", "pinion", "sprocket"],
            "required_all": ["rotational", "tooth_like"],
            "message": "Goal suggests toothed rotary geometry, but the graph lacks either a rotary blank/hub or a tooth-forming feature",
        },
        {
            "terms": ["shaft", "axle", "roller", "pulley", "bushing", "spacer"],
            "required_any": ["rotational", "revolved"],
            "message": "Goal suggests a rotational part, but the graph lacks cylindrical or revolved geometry",
        },
        {
            "terms": [
                "hole",
                "holes",
                "drill",
                "drilled",
                "bore",
                "slot",
                "cutout",
                "notch",
                "pocket",
                "window",
            ],
            "required_any": ["subtractive", "holes"],
            "message": "Goal requests removed material or openings, but the graph has no cut, pocket, bore, or hole feature",
        },
        {
            "terms": ["fillet", "rounded edge", "round edge", "rounded corner"],
            "required_all": ["rounded"],
            "message": "Goal requests rounded edges, but no fillet node is present",
        },
        {
            "terms": ["chamfer", "bevel", "beveled edge", "bevelled edge"],
            "required_all": ["beveled"],
            "message": "Goal requests beveled edges, but no chamfer node is present",
        },
        {
            "terms": ["shell", "hollow", "cavity", "hollowed"],
            "required_all": ["hollow"],
            "message": "Goal requests a hollow part, but the graph has no shelling or cavity-forming operation",
        },
        {
            "terms": ["plate", "bracket", "flange", "gusset", "mount"],
            "required_any": ["base_solid", "sketch_profile"],
            "message": "Goal suggests a plate or bracket-like part, but the graph lacks a clear base solid or profile",
        },
        {
            "terms": ["revolve", "lathe", "turned"],
            "required_all": ["revolved"],
            "message": "Goal explicitly suggests a revolved or turned shape, but no revolve node is present",
        },
        {
            "terms": ["sweep", "pipe", "tube", "handle", "rail"],
            "required_any": ["swept", "hollow", "rotational"],
            "message": "Goal suggests path-based or tubular geometry, but the graph lacks a sweep, tube, or hollow/rotational construction",
        },
        {
            "terms": ["loft", "blend", "transition"],
            "required_all": ["lofted"],
            "message": "Goal suggests a lofted transition, but no loft node is present",
        },
    ]

    for rule in semantic_rules:
        if not _has_goal_term(lowered_goal, *rule["terms"]):
            continue
        if _missing_features(
            features,
            required_any=rule.get("required_any"),
            required_all=rule.get("required_all"),
        ):
            issues.append(rule["message"])

    return issues


def _append_connection_if_missing(
    connections: list[dict[str, Any]], from_ref: str, to_ref: str
) -> bool:
    for connection in connections:
        if connection.get("from") == from_ref and connection.get("to") == to_ref:
            return False
    connections.append({"from": from_ref, "to": to_ref})
    return True


def _repair_cad_graph(data: dict[str, Any]) -> list[str]:
    """Apply deterministic repairs for common CAD graph failures."""
    repairs: list[str] = []
    nodes = data.get("nodes", [])
    connections = data.setdefault("connections", data.get("connections", []))
    node_order = {
        node.get("id", ""): index
        for index, node in enumerate(nodes)
        if isinstance(node, dict)
    }

    sketch_node_ids = [
        node.get("id", "")
        for node in nodes
        if isinstance(node, dict) and node.get("type") == "com.cad.sketch"
    ]
    sketch_profile_types = {
        "com.cad.sketch.line",
        "com.cad.sketch.circle",
        "com.cad.sketch.rectangle",
        "com.cad.sketch.polygon",
        "com.cad.sketch.arc",
        "com.cad.ellipse",
        "com.cad.spline",
        "com.cad.polyline",
    }

    if len(sketch_node_ids) == 1:
        sketch_source = f"{sketch_node_ids[0]}.sketch"
        for node in nodes:
            if not isinstance(node, dict):
                continue
            nid = node.get("id", "")
            ntype = node.get("type", "")
            schema = CAD_NODE_TYPES.get(ntype, {})
            if "sketch" not in schema.get("inputs", []):
                continue
            target = f"{nid}.sketch"
            if not any(conn.get("to") == target for conn in connections):
                if _append_connection_if_missing(connections, sketch_source, target):
                    repairs.append(f"connected lone sketch to '{nid}.sketch'")

    profile_candidates = [
        node.get("id", "")
        for node in nodes
        if isinstance(node, dict) and node.get("type") in sketch_profile_types
    ]
    shape_sources = [
        node.get("id", "")
        for node in nodes
        if isinstance(node, dict)
        and "shape" in CAD_NODE_TYPES.get(node.get("type", ""), {}).get("outputs", [])
        and node.get("type") != "com.cad.sketch"
    ]

    for node in nodes:
        if not isinstance(node, dict):
            continue
        nid = node.get("id", "")
        ntype = node.get("type", "")

        if (
            ntype in {"com.cad.extrude", "com.cad.twisted_extrude", "com.cad.revolve"}
            and len(profile_candidates) == 1
        ):
            input_port = "profile" if ntype == "com.cad.revolve" else "shape"
            target = f"{nid}.{input_port}"
            if not any(conn.get("to") == target for conn in connections):
                source = f"{profile_candidates[0]}.shape"
                if _append_connection_if_missing(connections, source, target):
                    repairs.append(
                        f"connected sketch profile '{profile_candidates[0]}' to '{nid}.{input_port}'"
                    )

        if ntype == "com.cad.geometry.boolean":
            current_index = node_order.get(nid, len(nodes))
            candidates = [
                source_id
                for source_id in shape_sources
                if source_id != nid and node_order.get(source_id, -1) < current_index
            ]
            used_sources = {
                conn.get("from", "").split(".", 1)[0]
                for conn in connections
                if conn.get("to", "").startswith(f"{nid}.")
            }
            available_sources = [
                source_id for source_id in candidates if source_id not in used_sources
            ]
            for input_port in ("base", "tool"):
                target = f"{nid}.{input_port}"
                if any(conn.get("to") == target for conn in connections):
                    continue
                if not available_sources:
                    break
                source_id = available_sources.pop()
                if _append_connection_if_missing(
                    connections, f"{source_id}.shape", target
                ):
                    repairs.append(
                        f"connected '{source_id}.shape' to missing boolean input '{nid}.{input_port}'"
                    )

    return repairs


def _sanitize_cad_params(data: dict) -> dict:
    """Strip hallucinated node types and properties before dispatch.

    The LLM sometimes invents properties that don't exist on a node.
    Rather than letting the dispatcher silently ignore them (or crash),
    we log a warning and remove them so the rest of the graph still
    works.  Unknown node types are also flagged.
    """
    data = normalize_node_specs(data)

    # Sketch-element types that NEED the 'sketch.' sub-prefix:
    #   sketch.circle, sketch.rectangle, sketch.polygon, sketch.arc, sketch.line
    # Sketch-element types WITHOUT 'sketch.' sub-prefix:
    #   polyline, spline, ellipse
    _SHORT_FIXES = {
        # Current GUI-native Design Studio geometry.
        "box": "geometry.box",
        "cylinder": "geometry.cylinder",
        "tube": "geometry.tube",
        "cylindrical_shell": "geometry.cylindrical_shell",
        "boolean": "geometry.boolean",
        "through_hole": "geometry.through_hole",
        "fillet": "geometry.fillet",
        "transform": "geometry.transform",
        "linear_pattern": "geometry.linear_pattern",
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
    _PHANTOM_TYPES = {
        "com.cad.parameter",
        "com.cad.reference",
        "com.cad.proxy",
        "com.cad.input",
        "com.cad.output",
        "com.cad.ref",
    }
    nodes = data.get("nodes", [])
    conns = data.get("connections", [])

    # Identify phantom nodes and build a bypass map:
    # phantom_id.port  →  the upstream node.port that feeds into it.
    phantom_ids: set = set()
    bypass_map: dict[str, str] = {}  # "phantom.shape" → "real_upstream.shape"
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
            internal_keys = {
                "type_",
                "name",
                "visible",
                "layout_direction",
                "subgraph_session",
                "selected",
                "disabled",
                "id",
                "icon",
            }
            for k in list(props.keys()):
                if k in internal_keys:
                    del props[k]
            props.update(custom)
            node["properties"] = props
            logger.info(
                f"Unwrapped nested 'custom' properties for node '{node.get('id', '?')}'"
            )

        # Strip class-name suffix: com.cad.box.BoxNode → com.cad.box
        # The LLM sometimes copies full class paths from the graph state.
        if ntype and re.search(r"\.[A-Z][a-zA-Z]*Node$", ntype):
            ntype = re.sub(r"\.[A-Z][a-zA-Z]*Node$", "", ntype)
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
            # Rescue: convert any unknown type to com.cad.code_part with
            # a sensible default box so at least something appears in the GUI.
            # The user can edit the code afterward.
            raw_props = node.get("properties", {})

            def _dim(keys: list[str], default: float) -> float:
                for k in keys:
                    if k in raw_props:
                        try:
                            return float(raw_props[k])
                        except (TypeError, ValueError):
                            continue
                return default

            t = ntype.lower()
            if "cylinder" in t or "cyl" in t:
                R = _dim(["radius", "r", "R", "width", "w"], 5.0)
                H = _dim(["height", "h", "H", "length", "l", "L"], 10.0)
                code = "result = cq.Workplane('XY').cylinder(H, R)"
                params = f"R={R}\nH={H}"
            elif "sphere" in t or "ball" in t:
                R = _dim(["radius", "r", "R", "size", "width"], 5.0)
                code = "result = cq.Workplane('XY').sphere(R)"
                params = f"R={R}"
            elif "cone" in t:
                R = _dim(["radius", "r", "bottom_radius", "width"], 10.0)
                H = _dim(["height", "h", "H", "length"], 20.0)
                code = "result = cq.Workplane('XY').newObject([cq.Solid.makeCone(R, 0, H)])"
                params = f"R={R}\nH={H}"
            else:
                # Default: box (covers box, cube, rect, primitive, custom_block, etc.)
                L = _dim(["length", "l", "L", "width", "w", "W", "size", "x"], 10.0)
                W = _dim(["width", "w", "W", "depth", "d", "D", "y"], 10.0)
                H = _dim(
                    ["height", "h", "H", "depth", "d", "D", "z", "thickness"], 10.0
                )
                code = "result = cq.Workplane('XY').box(L, W, H)"
                params = f"L={L}\nW={W}\nH={H}"
            node["type"] = "com.cad.code_part"
            node["properties"] = {"code": code, "parameters": params}
            logger.info(f"Normalized unknown type '{ntype}' → com.cad.code_part")
            continue
        valid_props = set(schema.get("properties", {}).keys())
        if not valid_props:
            continue
        props = node.get("properties", {})

        # --- Remap commonly hallucinated property names ----------------
        # The LLM frequently invents synonyms for real properties.
        # Map them to the correct names BEFORE the stripping step.
        _PROP_ALIASES: dict[str, dict[str, str]] = {
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
                    f"Remapped hallucinated props for {ntype}: {', '.join(remapped)}"
                )

        bad = [
            k
            for k in props
            if k not in valid_props and k not in ("center_x", "center_y", "center_z")
        ]  # center_* common to all
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


def _verify_cad_graph(data: dict, goal: str = "") -> list[str]:
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
    issues: list[str] = []

    if not nodes:
        return ["Empty node list — nothing to build"]

    # No primitive nodes exist — only com.cad.code_part for geometry.
    # Default-value checking is not applicable to code_part (free-form code).
    default_sensitive_nodes = {
        "com.cad.geometry.box",
        "com.cad.geometry.cylinder",
        "com.cad.geometry.tube",
        "com.cad.geometry.cylindrical_shell",
    }

    node_ids = set()
    node_map: dict[str, dict] = {}
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
        if ntype == "com.cad.geometry.boolean":
            a_connected = any(c.get("to", "") == f"{nid}.base" for c in conns)
            b_connected = any(c.get("to", "") == f"{nid}.tool" for c in conns)
            if not a_connected:
                issues.append(f"Boolean '{nid}' missing base connection")
            if not b_connected:
                issues.append(f"Boolean '{nid}' missing tool connection")

        # Numeric properties all at default — LLM didn't customize dims
        defaults = schema.get("properties", {})
        if defaults and props and ntype in default_sensitive_nodes:
            # Only check numeric properties (dimensions, counts, angles).
            # String/enum props (like operation='Union') are intentional.
            # Exclude bool because in Python bool is a subclass of int.
            # Skip nodes that have NO numeric defaults (e.g. loft with only 'ruled').
            numeric_defaults = {
                k: v
                for k, v in defaults.items()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and k in props
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

    issues.extend(_verify_cad_semantics(data, goal))
    return issues


def run_cad_verified(
    data: dict[str, Any],
    dispatcher: "CommandDispatcher",
) -> dict[str, Any]:
    """Sanitize, verify, and dispatch a CAD graph with structured diagnostics."""
    goal = str(data.get("goal", "") or "")
    raw_payload = deepcopy(data)
    data = _sanitize_cad_params(data)
    applied_repairs = _repair_cad_graph(data)
    _log_graph_payload("normalized", data, goal)
    if raw_payload != data:
        _log_graph_payload("raw", raw_payload, goal)
    if applied_repairs:
        logger.info(f"CAD deterministic repairs applied: {applied_repairs}")
        _log_graph_payload("repaired", data, goal)
    issues = _verify_cad_graph(data, goal)
    if issues:
        for issue in issues:
            logger.warning(f"CAD graph issue: {issue}")
    # Dispatch even with warnings — let the engine handle what it can
    result = dispatcher._build_node_graph({"params": data}, sync=True)
    if not isinstance(result, dict):
        result = {
            "success": result is not None,
            "result": result,
        }
    if applied_repairs:
        result["applied_repairs"] = applied_repairs
    if issues:
        result["verification_warnings"] = issues
    return result


def verify_cad_graph(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize and verify CAD graph JSON without executing it."""
    goal = str(data.get("goal", "") or "")
    original = deepcopy(data)
    sanitized = _sanitize_cad_params(deepcopy(data))
    applied_repairs = _repair_cad_graph(sanitized)
    _log_graph_payload("verify_raw", original, goal)
    _log_graph_payload("verify_sanitized", sanitized, goal)
    if applied_repairs:
        _log_graph_payload("verify_repaired", sanitized, goal)
    issues = _verify_cad_graph(sanitized, goal)
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "applied_repairs": applied_repairs,
        "sanitized": sanitized,
    }
