# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Normalize, repair, and validate assistant-authored system-model graphs."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from pylcss.assistant_systems.tools.graph_normalization import normalize_node_specs

if TYPE_CHECKING:
    from pylcss.assistant_systems.api.dispatcher import CommandDispatcher

logger = logging.getLogger(__name__)


def _build_passthrough_code(num_inputs: int, num_outputs: int) -> str:
    input_names = [f"in_{i}" for i in range(1, max(num_inputs, 1) + 1)]
    lines = ["import numpy as np", ""]
    for output_index in range(1, max(num_outputs, 1) + 1):
        source_name = input_names[min(output_index - 1, len(input_names) - 1)]
        lines.append(f"out_{output_index} = {source_name}")
    return "\n".join(lines)


def _repair_system_graph(data: dict[str, Any]) -> list[str]:
    """Apply deterministic repairs for common system-model graph failures."""
    repairs: list[str] = []
    nodes = data.get("nodes", [])

    for node in nodes:
        if not isinstance(node, dict):
            continue
        nid = node.get("id", "?")
        ntype = node.get("type", "")
        props = node.get("properties", {})

        if ntype == "com.pfd.custom_block":
            code = str(props.get("code_content", ""))
            real_code = "\n".join(
                line
                for line in code.splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
            if not real_code.strip():
                try:
                    num_inputs = max(int(props.get("num_inputs", 1)), 1)
                except (TypeError, ValueError):
                    num_inputs = 1
                try:
                    num_outputs = max(int(props.get("num_outputs", 1)), 1)
                except (TypeError, ValueError):
                    num_outputs = 1
                props["code_content"] = _build_passthrough_code(num_inputs, num_outputs)
                repairs.append(
                    f"generated fallback pass-through code for CustomBlock '{nid}'"
                )

        if ntype == "com.pfd.input":
            try:
                min_value = float(props.get("min", 0))
                max_value = float(props.get("max", 10))
                if min_value >= max_value:
                    props["min"] = min(min_value, max_value)
                    props["max"] = max(min_value, max_value) + 1.0
                    repairs.append(f"adjusted invalid min/max bounds for Input '{nid}'")
            except (TypeError, ValueError):
                props["min"] = 0.0
                props["max"] = 10.0
                repairs.append(f"reset invalid min/max bounds for Input '{nid}'")

        if (
            ntype == "com.pfd.output"
            and props.get("minimize")
            and props.get("maximize")
        ):
            props["maximize"] = False
            repairs.append(f"cleared conflicting maximize flag for Output '{nid}'")

    return repairs


def _verify_system_graph(data: dict) -> list[str]:
    """Verify structural integrity of an LLM-generated system modeling graph.

    Returns a list of issue strings (empty = all good).
    Checks:
    1. Disconnected nodes
    2. CustomBlock with empty/commented-out code_content
    3. Input nodes with min >= max
    4. Output node with both minimize and maximize
    5. Connections referencing non-existent nodes
    """
    data = normalize_node_specs(data)
    nodes = data.get("nodes", [])
    conns = data.get("connections", [])
    issues: list[str] = []

    if not nodes:
        return ["Empty node list"]

    node_ids = set()
    node_map: dict[str, dict] = {}
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
                line
                for line in code.splitlines()
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
                issues.append(f"Input '{nid}' has non-numeric min/max bounds")

        # Output with both minimize and maximize
        if ntype == "com.pfd.output":
            if props.get("minimize") and props.get("maximize"):
                issues.append(
                    f"Output '{nid}': both minimize and maximize set — pick one"
                )

    return issues


def run_system_verified(
    data: dict[str, Any],
    dispatcher: "CommandDispatcher",
) -> dict[str, Any]:
    """Verify and dispatch a system-model graph with structured diagnostics."""
    data = normalize_node_specs(data)
    applied_repairs = _repair_system_graph(data)
    issues = _verify_system_graph(data)
    if issues:
        for issue in issues:
            logger.warning(f"System graph issue: {issue}")
    if applied_repairs:
        logger.info(f"System deterministic repairs applied: {applied_repairs}")
    result = dispatcher._build_system_graph({"params": data}, sync=True)
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


def verify_system_graph(data: dict[str, Any]) -> dict[str, Any]:
    """Verify system-model graph JSON without executing it."""
    checked = normalize_node_specs(deepcopy(data))
    applied_repairs = _repair_system_graph(checked)
    issues = _verify_system_graph(checked)
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "applied_repairs": applied_repairs,
        "sanitized": checked,
    }
