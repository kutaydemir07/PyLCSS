# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Bounded, thread-safe storage for completed study evaluations."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import sys
from threading import Lock
from typing import Any

import numpy as np


def result_size_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate retained Python/array memory without traversing CAD internals."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        if value.base is not None:
            return size + result_size_bytes(value.base, seen)
        return max(size, value.nbytes)
    if isinstance(value, Mapping):
        return size + sum(
            result_size_bytes(key, seen) + result_size_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(result_size_bytes(item, seen) for item in value)
    if hasattr(value, "p") and hasattr(value, "t"):
        return (
            size + result_size_bytes(value.p, seen) + result_size_bytes(value.t, seen)
        )
    return size


class EvaluationCache:
    """Least-recently-used cache with entry and estimated memory limits."""

    def __init__(self, max_entries: int = 32, max_bytes: int = 256 * 1024**2):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("Cache limits must be positive.")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[tuple, tuple[Any, int]] = OrderedDict()
        self._size_bytes = 0
        self._lock = Lock()

    def get(self, key: tuple) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: tuple, value: Any, size_bytes: int) -> None:
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._size_bytes -= previous[1]
            if size_bytes > self.max_bytes:
                return
            self._entries[key] = (value, size_bytes)
            self._size_bytes += size_bytes
            while (
                len(self._entries) > self.max_entries
                or self._size_bytes > self.max_bytes
            ):
                _, (_, removed_size) = self._entries.popitem(last=False)
                self._size_bytes -= removed_size

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._size_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._size_bytes
