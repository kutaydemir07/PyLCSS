# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded, single-generation reuse of immutable lattice clipping geometry."""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np


class LatticeGeometryCache:
    """Never shared across jobs; changed density/cell fields are not cached.

    Source CAD/STL objects must remain immutable during the generation. Keep
    source references alive to prevent Python object identities being reused.
    Arrays are copied on return because clipping operations mutate their inputs.
    """
    def __init__(self, maximum_bytes=512 * 1024**2):
        self.maximum_bytes = int(maximum_bytes)
        self.entries = {}
        self.sources = {}
        self.bytes = self.hits = self.misses = 0

    def source_key(self, source):
        self.sources[id(source)] = source
        return id(source)

    def get(self, key, factory):
        if key in self.entries:
            self.hits += 1
            return deepcopy(self.entries[key])
        self.misses += 1
        value = factory()
        size = self._size(value)
        if self.bytes + size <= self.maximum_bytes:
            self.entries[key] = deepcopy(value)
            self.bytes += size
        return value

    @staticmethod
    def _size(value):
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, (tuple, list)):
            return sum(LatticeGeometryCache._size(item) for item in value)
        if isinstance(value, dict):
            return sum(LatticeGeometryCache._size(item) for item in value.values())
        return 128

    def triangle_mesh(self, source, tolerance):
        from .geometry_lattice import geometry_triangle_mesh
        key = ('triangles', self.source_key(source), float(tolerance))
        def build():
            mesh = geometry_triangle_mesh(source, tolerance)
            return np.asarray(mesh.vertices).copy(), np.asarray(mesh.faces).copy()
        import trimesh
        vertices, faces = self.get(key, build)
        return trimesh.Trimesh(vertices, faces, process=False)

    @staticmethod
    def boundary_key(mesh, axes):
        digest = hashlib.sha256()
        for array in (mesh.vertices, mesh.faces, *axes):
            values = np.ascontiguousarray(array)
            digest.update(str((values.shape, values.dtype.str)).encode('ascii'))
            digest.update(values.tobytes())
        return 'distance', digest.hexdigest()

    def report(self):
        return dict(scope='This generation only; fixed clipping geometry, never changed lattice material',
                    hits=self.hits, misses=self.misses, cached_bytes=self.bytes,
                    maximum_bytes=self.maximum_bytes)
