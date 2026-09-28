# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Thread-safe caching for expensive CAD geometry evaluations."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

from .cad_geometry import CadGeometry, cad_evaluate_geometry


class GeometryCache:
    """Bounded least-recently-used cache keyed by graph state and parameters."""

    def __init__(self, max_entries: int = 64) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")
        self.max_entries = int(max_entries)
        self._store: OrderedDict[str, CadGeometry] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(
        cad_path: str,
        kind: str,
        params: Mapping[str, float],
        field_name: str | None = None,
    ) -> str:
        path = Path(cad_path).expanduser().resolve(strict=False)
        try:
            stat = path.stat()
            path_identity = f"{path}|{stat.st_mtime_ns}|{stat.st_size}"
        except OSError:
            path_identity = str(path)

        digest = hashlib.sha1()
        for part in (path_identity, kind, field_name or "*"):
            digest.update(part.encode("utf-8"))
            digest.update(b"|")
        for name, value in sorted(
            (name, float(value)) for name, value in params.items()
        ):
            digest.update(f"{name}={value:.12g};".encode())
        return digest.hexdigest()

    def get(
        self,
        cad_path: str,
        kind: str,
        params: Mapping[str, float],
        field_name: str | None = None,
    ) -> CadGeometry | None:
        key = self._key(cad_path, kind, params, field_name)
        with self._lock:
            geometry = self._store.get(key)
            if geometry is not None:
                self._store.move_to_end(key)
            return geometry

    def put(
        self,
        cad_path: str,
        kind: str,
        params: Mapping[str, float],
        geometry: CadGeometry,
        field_name: str | None = None,
    ) -> None:
        key = self._key(cad_path, kind, params, field_name)
        with self._lock:
            if key in self._store:
                self._store[key] = geometry
                self._store.move_to_end(key)
                return
            while len(self._store) >= self.max_entries:
                self._store.popitem(last=False)
            self._store[key] = geometry

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_DEFAULT_CACHE = GeometryCache(max_entries=128)


def evaluate_with_cache(
    cad_path: str,
    kind: str,
    params: Mapping[str, float],
    field_name: str | None = None,
    cache: GeometryCache | None = None,
) -> CadGeometry:
    """Evaluate a CAD graph once per file, solver, field, and parameter set."""
    selected_cache = cache if cache is not None else _DEFAULT_CACHE
    cached = selected_cache.get(cad_path, kind, params, field_name)
    if cached is not None:
        return cached
    geometry = cad_evaluate_geometry(cad_path, kind, params, field_name=field_name)
    selected_cache.put(cad_path, kind, params, geometry, field_name)
    return geometry


__all__ = ["GeometryCache", "evaluate_with_cache"]
