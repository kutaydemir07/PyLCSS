# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Safe, portable persistence for Design Studio simulation results.

The graph remains human-readable JSON. Numerical FEA, crash, and topology
results are stored in a compressed HDF5 sidecar instead of pickling live Python
objects. Unsupported runtime objects (OCC shapes, solver handles, callbacks)
are deliberately skipped.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote

import numpy as np

logger = logging.getLogger(__name__)

FORMAT_NAME = "pylcss-design-results"
FORMAT_VERSION = 1
RESULT_TYPES = {"fea", "crash", "topopt_voxel", "remesh"}
_SKIPPED_RUNTIME_KEYS = {
    "shape",
    "cad",
    "cad_shape",
    "source_shape",
    "geometry",
    "solver",
    "basis",
}


class PersistedMesh:
    """Minimal scikit-fem-compatible mesh used for result visualization."""

    def __init__(self, p: Any, t: Any):
        self.p = np.asarray(p, dtype=float)
        self.t = np.asarray(t, dtype=int)


def _safe_name(value: Any) -> str:
    return quote(str(value), safe="")


def _display_node_name(node: Any) -> str:
    name = getattr(node, "name", None)
    if callable(name):
        name = name()
    return str(name or getattr(node, "NODE_NAME", None) or node.__class__.__name__)


def _write_dataset(parent: Any, name: str, value: np.ndarray, kind: str) -> None:
    array = np.asarray(value)
    kwargs = {}
    if array.ndim and array.size > 16:
        kwargs = {"compression": "gzip", "shuffle": True}
    dataset = parent.create_dataset(name, data=array, **kwargs)
    dataset.attrs["_kind"] = kind


def _write_value(
    parent: Any,
    key: str,
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> bool:
    """Write a supported value. Return False for runtime-only objects."""
    if depth > 12:
        return False
    seen = seen if seen is not None else set()
    name = _safe_name(key)

    if value is None:
        group = parent.create_group(name)
        group.attrs["_kind"] = "none"
        return True
    if isinstance(value, (bool, int, float, np.number)):
        _write_dataset(parent, name, np.asarray(value), "scalar")
        return True
    if isinstance(value, str):
        import h5py

        dataset = parent.create_dataset(
            name, data=value, dtype=h5py.string_dtype(encoding="utf-8")
        )
        dataset.attrs["_kind"] = "string"
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return False
        if value.dtype.kind in {"U", "S"}:
            import h5py

            data = np.asarray(value, dtype=object)
            dataset = parent.create_dataset(
                name, data=data, dtype=h5py.string_dtype(encoding="utf-8")
            )
            dataset.attrs["_kind"] = "ndarray_string"
            return True
        _write_dataset(parent, name, value, "ndarray")
        return True

    object_id = id(value)
    if object_id in seen:
        return False

    if hasattr(value, "p") and hasattr(value, "t"):
        seen.add(object_id)
        group = parent.create_group(name)
        group.attrs["_kind"] = "mesh"
        _write_value(group, "p", np.asarray(value.p), depth=depth + 1, seen=seen)
        _write_value(group, "t", np.asarray(value.t), depth=depth + 1, seen=seen)
        return True

    if isinstance(value, dict):
        seen.add(object_id)
        group = parent.create_group(name)
        group.attrs["_kind"] = "dict"
        for child_key, child_value in value.items():
            child_key = str(child_key)
            if child_key in _SKIPPED_RUNTIME_KEYS:
                continue
            try:
                _write_value(
                    group,
                    child_key,
                    child_value,
                    depth=depth + 1,
                    seen=seen,
                )
            except Exception:
                logger.debug(
                    "Skipping unsupported result value %s", child_key,
                    exc_info=True,
                )
        return True

    if isinstance(value, (list, tuple)):
        seen.add(object_id)
        try:
            array = np.asarray(value)
        except Exception:
            array = np.asarray([], dtype=object)
        if array.dtype != object and array.dtype.kind not in {"U", "S"}:
            _write_dataset(
                parent,
                name,
                array,
                "tuple_array" if isinstance(value, tuple) else "list_array",
            )
            return True
        group = parent.create_group(name)
        group.attrs["_kind"] = "tuple" if isinstance(value, tuple) else "list"
        for index, child_value in enumerate(value):
            _write_value(
                group,
                str(index),
                child_value,
                depth=depth + 1,
                seen=seen,
            )
        return True

    return False


def _read_value(item: Any) -> Any:
    import h5py

    kind = item.attrs.get("_kind", "")
    if isinstance(kind, bytes):
        kind = kind.decode("utf-8")
    if isinstance(item, h5py.Dataset):
        value = item[()]
        if kind == "string":
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if kind == "ndarray_string":
            return np.asarray([
                entry.decode("utf-8") if isinstance(entry, bytes) else str(entry)
                for entry in np.asarray(value).flat
            ]).reshape(np.asarray(value).shape)
        if kind == "scalar":
            return value.item() if isinstance(value, np.generic) else value
        if kind == "list_array":
            return np.asarray(value).tolist()
        if kind == "tuple_array":
            return tuple(np.asarray(value).tolist())
        return np.asarray(value)

    if kind == "none":
        return None
    if kind == "mesh":
        return PersistedMesh(
            _read_value(item[_safe_name("p")]),
            _read_value(item[_safe_name("t")]),
        )
    if kind == "dict":
        return {unquote(name): _read_value(child) for name, child in item.items()}
    if kind in {"list", "tuple"}:
        values = [
            _read_value(item[name])
            for name in sorted(item.keys(), key=lambda entry: int(unquote(entry)))
        ]
        return tuple(values) if kind == "tuple" else values
    return {unquote(name): _read_value(child) for name, child in item.items()}


def save_results(path: str | os.PathLike[str], nodes: Iterable[Any]) -> int:
    """Atomically save every cached simulation result and return its count."""
    import h5py

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    count = 0
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = FORMAT_NAME
            handle.attrs["version"] = FORMAT_VERSION
            node_group = handle.create_group("nodes")
            for node in nodes:
                result = getattr(node, "_last_result", None)
                if not isinstance(result, dict):
                    continue
                if str(result.get("type") or "") not in RESULT_TYPES:
                    continue
                node_id = str(getattr(node, "id", "") or "")
                if not node_id:
                    continue
                group = node_group.create_group(_safe_name(node_id))
                group.attrs["node_id"] = node_id
                group.attrs["node_name"] = _display_node_name(node)
                group.attrs["identifier"] = str(
                    getattr(node, "__identifier__", "") or ""
                )
                if _write_value(group, "result", result):
                    count += 1
            handle.attrs["result_count"] = count
            handle.flush()
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return count


def load_results(path: str | os.PathLike[str], nodes: Iterable[Any]) -> int:
    """Restore cached results by stable NodeGraphQt node id."""
    import h5py

    by_id = {
        str(getattr(node, "id", "") or ""): node
        for node in nodes
        if getattr(node, "id", None)
    }
    restored = 0
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("format", "")) != FORMAT_NAME:
            raise ValueError("Unrecognized Design Studio results sidecar.")
        if int(handle.attrs.get("version", 0)) != FORMAT_VERSION:
            raise ValueError("Unsupported Design Studio results sidecar version.")
        for group in handle.get("nodes", {}).values():
            node_id = str(group.attrs.get("node_id", ""))
            node = by_id.get(node_id)
            result_key = _safe_name("result")
            if node is None or result_key not in group:
                continue
            result = _read_value(group[result_key])
            if not isinstance(result, dict):
                continue
            node._last_result = result
            node._last_input_hash = None
            node._persisted_result = True
            node._result_revision = (
                int(getattr(node, "_result_revision", 0)) + 1
            )
            restored += 1
    return restored
