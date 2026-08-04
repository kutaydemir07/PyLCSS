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
from types import SimpleNamespace
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

# Result objects that are restored by name rather than skipped. This is a
# whitelist, not generic class resolution: a saved file can only ever name one
# of these, and each is a plain frozen dataclass of scalars that validates its
# own fields on construction. Nothing else in a sidecar is instantiated.
#
# `ManufacturingStructureOptions` is what tells a reopened study that its result
# is a lattice rather than a solid. Without it the viewer offered the solid CAD
# view for a restored lattice, found no B-rep behind it — the live shape is a
# runtime object and is deliberately not persisted — and drew nothing, while the
# density field it was built from still rendered.
# `OptimizedMemberPlan` carries the sized strut network: the centrelines and
# the per-member radii the sizing pass solved for. Losing it made a reopened
# strut study display and export members at their nominal thickness instead of
# the ones it was analysed with.
_RESTORABLE_DATACLASS_KIND = "dataclass"
_RESTORABLE_DATACLASSES = {
    "ManufacturingStructureOptions": (
        "pylcss.design_studio.topology_optimization.manufacturing.structures",
        "ManufacturingStructureOptions",
    ),
    "OptimizedMemberPlan": (
        "pylcss.design_studio.topology_optimization.manufacturing.member_sizing",
        "OptimizedMemberPlan",
    ),
}


def _restorable_dataclass_name(value: Any) -> str | None:
    """Return the whitelisted dataclass name for *value*, if it is one."""
    import dataclasses

    name = type(value).__name__
    if name in _RESTORABLE_DATACLASSES and dataclasses.is_dataclass(value):
        return name
    return None


def _restore_dataclass(name: str, values: dict[str, Any]) -> Any:
    """Rebuild a whitelisted dataclass, or return its fields on failure."""
    import dataclasses
    import importlib

    target = _RESTORABLE_DATACLASSES.get(str(name))
    if target is None:
        return values
    try:
        cls = getattr(importlib.import_module(target[0]), target[1])
        accepted = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in accepted})
    except Exception:
        # A result written by a newer build, or one whose validation rules have
        # since tightened, must not make the project impossible to open.
        logger.warning(
            "Could not restore saved %s from the result sidecar.", name,
            exc_info=True,
        )
        return None


class PersistedMesh:
    """Minimal scikit-fem-compatible mesh used for result visualization."""

    def __init__(self, p: Any, t: Any, element_dofs: Any = None):
        self.p = np.asarray(p, dtype=float)
        self.t = np.asarray(t, dtype=int)
        if element_dofs is not None:
            self.dofs = SimpleNamespace(
                element_dofs=np.asarray(element_dofs, dtype=int)
            )


def _safe_name(value: Any) -> str:
    return quote(str(value), safe="")


def _display_node_name(node: Any) -> str:
    name = getattr(node, "name", None)
    if callable(name):
        name = name()
    return str(name or getattr(node, "NODE_NAME", None) or node.__class__.__name__)


def _node_identifier(node: Any) -> str:
    return str(getattr(node, "__identifier__", "") or "")


def _attribute_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value or "")


# NodeGraphQt regenerates node ids during deserialization, so a sidecar is
# matched to its study by identifier and display name. Lattice optimization
# used to be a mode of the topology node: a project saved before the split
# records the topology identifier for a study that now loads as the lattice
# node, and matching those two literally would silently restore nothing. They
# are one identity here so an existing result keeps loading.
def _identity_key(identifier: Any, name: Any) -> tuple[str, str]:
    """Return the save-stable identity used to match a result to a node."""
    from pylcss.design_studio.topology_optimization.integration.study_identity import (
        DENSITY_STUDY_IDENTIFIERS,
    )

    text = str(identifier or "")
    if text in DENSITY_STUDY_IDENTIFIERS:
        text = "com.cad.sim.density_study"
    return (text, str(name or ""))


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

    dataclass_name = _restorable_dataclass_name(value)
    if dataclass_name is not None:
        import dataclasses

        seen.add(object_id)
        group = parent.create_group(name)
        group.attrs["_kind"] = _RESTORABLE_DATACLASS_KIND
        group.attrs["_dataclass"] = dataclass_name
        for field in dataclasses.fields(value):
            _write_value(
                group,
                field.name,
                getattr(value, field.name),
                depth=depth + 1,
                seen=seen,
            )
        return True

    if hasattr(value, "p") and hasattr(value, "t"):
        seen.add(object_id)
        group = parent.create_group(name)
        group.attrs["_kind"] = "mesh"
        _write_value(group, "p", np.asarray(value.p), depth=depth + 1, seen=seen)
        _write_value(group, "t", np.asarray(value.t), depth=depth + 1, seen=seen)
        element_dofs = getattr(getattr(value, "dofs", None), "element_dofs", None)
        if element_dofs is not None:
            _write_value(
                group,
                "element_dofs",
                np.asarray(element_dofs),
                depth=depth + 1,
                seen=seen,
            )
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
            (
                _read_value(item[_safe_name("element_dofs")])
                if _safe_name("element_dofs") in item
                else None
            ),
        )
    if kind == _RESTORABLE_DATACLASS_KIND:
        dataclass_name = item.attrs.get("_dataclass", "")
        if isinstance(dataclass_name, bytes):
            dataclass_name = dataclass_name.decode("utf-8")
        return _restore_dataclass(
            str(dataclass_name),
            {unquote(name): _read_value(child) for name, child in item.items()},
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


def save_results(
    path: str | os.PathLike[str],
    nodes: Iterable[Any],
    progress_callback=None,
) -> int:
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
                is_mesh = hasattr(result, "p") and hasattr(result, "t")
                if not is_mesh and (
                    not isinstance(result, dict)
                    or str(result.get("type") or "") not in RESULT_TYPES
                ):
                    continue
                node_id = str(getattr(node, "id", "") or "")
                if not node_id:
                    continue
                group = node_group.create_group(_safe_name(node_id))
                group.attrs["node_id"] = node_id
                group.attrs["node_name"] = _display_node_name(node)
                group.attrs["identifier"] = _node_identifier(node)
                if _write_value(group, "result", result):
                    count += 1
                if progress_callback is not None:
                    progress_callback()
            handle.attrs["result_count"] = count
            handle.flush()
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return count


def load_results(
    path: str | os.PathLike[str],
    nodes: Iterable[Any],
    progress_callback=None,
) -> int:
    """Restore cached results by node id or unique type/name identity.

    NodeGraphQt versions that regenerate runtime ids during deserialization
    cannot match a sidecar by id alone.  The saved identifier and display name
    provide a deterministic fallback for hand-authored example graphs while
    still refusing ambiguous matches.
    """
    import h5py

    node_list = list(nodes)
    by_id = {
        str(getattr(node, "id", "") or ""): node
        for node in node_list
        if getattr(node, "id", None)
    }
    by_identity: dict[tuple[str, str], list[Any]] = {}
    for node in node_list:
        identity = _identity_key(
            _node_identifier(node), _display_node_name(node)
        )
        by_identity.setdefault(identity, []).append(node)

    restored = 0
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("format", "")) != FORMAT_NAME:
            raise ValueError("Unrecognized Design Studio results sidecar.")
        if int(handle.attrs.get("version", 0)) != FORMAT_VERSION:
            raise ValueError("Unsupported Design Studio results sidecar version.")
        for group in handle.get("nodes", {}).values():
            node_id = _attribute_text(group.attrs.get("node_id", ""))
            node = by_id.get(node_id)
            if node is None:
                identity = _identity_key(
                    _attribute_text(group.attrs.get("identifier", "")),
                    _attribute_text(group.attrs.get("node_name", "")),
                )
                candidates = by_identity.get(identity, ())
                if len(candidates) == 1:
                    node = candidates[0]
            result_key = _safe_name("result")
            if node is None or result_key not in group:
                continue
            result = _read_value(group[result_key])
            if not isinstance(result, dict) and not (
                hasattr(result, "p") and hasattr(result, "t")
            ):
                continue
            node._last_result = result
            node._last_input_hash = None
            node._dirty = False
            node._force_execute = False
            node._persisted_result = True
            node._result_revision = (
                int(getattr(node, "_result_revision", 0)) + 1
            )
            restored += 1
            if progress_callback is not None:
                progress_callback()
    return restored
