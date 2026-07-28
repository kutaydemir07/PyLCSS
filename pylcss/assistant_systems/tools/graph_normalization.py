# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Shared normalization for assistant-authored graph payloads."""

from __future__ import annotations

from typing import Any


def normalize_node_specs(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize flat LLM node specs into {id, type, properties} form.

    Models sometimes emit node properties at the top level instead of inside
    a nested ``properties`` object. Normalize that shape before verification
    and dispatch so the downstream graph builders interpret the request
    correctly.
    """
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        nodes = []
    connections = data.get("connections", [])
    if not isinstance(connections, list):
        data["connections"] = []
    normalized_nodes = []

    for node in nodes:
        if not isinstance(node, dict):
            normalized_nodes.append(node)
            continue

        normalized = dict(node)
        props = normalized.get("properties", {})
        if not isinstance(props, dict):
            props = {}

        flat_props = {
            key: value
            for key, value in normalized.items()
            if key not in ("id", "type", "properties")
        }
        if flat_props:
            merged_props = dict(flat_props)
            merged_props.update(props)
            normalized = {
                "id": normalized.get("id"),
                "type": normalized.get("type"),
                "properties": merged_props,
            }

        normalized_nodes.append(normalized)

    data["nodes"] = normalized_nodes
    return data
