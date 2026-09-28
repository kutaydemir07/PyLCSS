# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Crash-safe persistence for system-modeling graphs."""

from __future__ import annotations

from os import PathLike
from typing import Any

from pylcss.io_manager.project_io import atomic_json_dump, load_json_object

FORMAT_VERSION = 1


def save_systems(
    manager: Any,
    path: str | PathLike[str],
) -> None:
    """Atomically serialize every graph owned by *manager*."""

    product_name = manager.product_name.text().strip() or "Product"
    systems = []
    for index, system in enumerate(manager.systems):
        name = system.get("name")
        graph = system.get("graph")
        if not isinstance(name, str) or not name.strip() or graph is None:
            raise ValueError(f"System {index} is incomplete and cannot be saved.")
        systems.append(
            {
                "name": name.strip(),
                "graph": graph.serialize_session(),
            }
        )
    atomic_json_dump(
        {
            "_copyright": "Copyright (c) 2026 Kutay Demir.",
            "_license": (
                "Licensed under the PolyForm Shield License 1.0.0. "
                "See LICENSE file for details."
            ),
            "format_version": FORMAT_VERSION,
            "product_name": product_name,
            "systems": systems,
        },
        path,
    )


def load_systems(
    manager: Any,
    path: str | PathLike[str],
) -> None:
    """Deserialize all graphs, committing them only after every graph succeeds."""

    data = load_json_object(path, required_keys=("systems",))
    records = _validate_document(data)
    prepared: list[tuple[str, Any]] = []
    try:
        for record in records:
            graph = manager.create_graph()
            try:
                graph.deserialize_session(record["graph"])
                manager.prepare_loaded_graph(graph)
            except Exception:
                graph.widget.deleteLater()
                raise
            prepared.append((record["name"], graph))
    except Exception:
        for _name, graph in prepared:
            graph.widget.deleteLater()
        raise

    manager.replace_systems(prepared)
    product_name = data.get("product_name", "Product")
    manager.product_name.setText(product_name.strip() or "Product")


def _validate_document(data: dict[str, Any]) -> list[dict[str, Any]]:
    version = data.get("format_version", 1)
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported system-model format {version!r}; "
            f"expected {FORMAT_VERSION}."
        )
    product_name = data.get("product_name", "Product")
    if not isinstance(product_name, str):
        raise ValueError("'product_name' must be a string.")

    raw_systems = data["systems"]
    if not isinstance(raw_systems, list):
        raise ValueError("'systems' must be a list.")
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, system in enumerate(raw_systems):
        if not isinstance(system, dict):
            raise ValueError(f"System {index} must be an object.")
        name = system.get("name")
        session = system.get("graph")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"System {index} has no valid name.")
        normalized_name = name.strip()
        if normalized_name in names:
            raise ValueError(f"Duplicate system name: {normalized_name!r}.")
        if not isinstance(session, dict):
            raise ValueError(f"System {index} has no valid graph object.")
        names.add(normalized_name)
        records.append({"name": normalized_name, "graph": session})
    return records


__all__ = ["FORMAT_VERSION", "load_systems", "save_systems"]
