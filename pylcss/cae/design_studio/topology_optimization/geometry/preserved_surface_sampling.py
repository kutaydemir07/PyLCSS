# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Exact sampled operands bound to the CAD state that produced a certificate."""
from __future__ import annotations

import numpy as np

from .amrto_surface import geometry_fingerprint
from .cad_fingerprint import cad_shape_fingerprint


class PreservedSurfaceOperand:
    """Retain one checked CAD tessellation across trusted CAD job transfer.

    BREP import can renormalize direction coefficients without changing the
    transferred geometry. Validate the exact original state before pickling,
    then bind the retained mesh to the actual deserialized CAD object. This
    avoids a rounded geometry hash and retains sensitivity to subsequent CAD
    edits, including topology, placement, orientation and tolerance changes.
    The caller must still rerun every material/void/source mesh certificate.
    """

    def __init__(self, shape, vertices, faces):
        self._shape = shape
        self._vertices, self._faces = self._arrays(vertices, faces)
        self._cad_fingerprint = cad_shape_fingerprint(shape)
        self._mesh_fingerprint = geometry_fingerprint(self._vertices, self._faces)

    @staticmethod
    def _arrays(vertices, faces):
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or not np.isfinite(vertices).all() or faces.ndim != 2
                or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Preserved sampling requires finite indexed triangle geometry')
        vertices = np.array(vertices, dtype=np.float64, order='C', copy=True)
        faces = np.array(faces, dtype=np.int64, order='C', copy=True)
        vertices.flags.writeable = faces.flags.writeable = False
        return vertices, faces

    def _validate_retained_state(self):
        if geometry_fingerprint(self._vertices, self._faces) != self._mesh_fingerprint:
            raise ValueError('Preserved sampling triangles changed; regenerate the preserved surface')
        if cad_shape_fingerprint(self._shape) != self._cad_fingerprint:
            raise ValueError('Preserved sampling CAD state changed; regenerate the preserved surface')

    def validated_mesh(self, current_shape=None):
        """Return isolated arrays after retained and optional current CAD checks.

        Derived tools use their exact upstream input provenance after transfer;
        rebuilding a cylinder from an imported direction can change its last
        coefficient bits. Their retained shape and triangles remain checked.
        """
        self._validate_retained_state()
        if current_shape is not None and cad_shape_fingerprint(current_shape) != self._cad_fingerprint:
            raise ValueError('Preserved sampling does not match the current CAD state; '
                             'regenerate the preserved surface')
        return self._vertices.copy(), self._faces.copy()

    def __getstate__(self):
        # This executes before the private CAD job pickler writes the referenced
        # shape. A CAD edit or altered operand cannot be legitimized by transfer.
        self._validate_retained_state()
        return {'version': 1, 'shape': self._shape, 'vertices': self._vertices,
                'faces': self._faces, 'mesh_fingerprint': self._mesh_fingerprint}

    def __setstate__(self, state):
        if state.get('version') != 1:
            raise ValueError('Unsupported preserved sampling transfer version')
        vertices, faces = self._arrays(state['vertices'], state['faces'])
        if geometry_fingerprint(vertices, faces) != state['mesh_fingerprint']:
            raise ValueError('Preserved sampling triangles changed during CAD transfer')
        self._shape, self._vertices, self._faces = state['shape'], vertices, faces
        self._mesh_fingerprint = state['mesh_fingerprint']
        # The trusted transfer has already deserialized this same CAD reference.
        # Bind its exact current representation; do not quantize any coefficients.
        self._cad_fingerprint = cad_shape_fingerprint(self._shape)


def _freeze_definition(value):
    """Snapshot constraint values without rounding or retaining mutable arrays."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return ('array_objects', value.shape, _freeze_definition(value.tolist()))
        return ('array', value.dtype.str, value.shape, value.tobytes(order='C'))
    if isinstance(value, np.generic):
        return ('scalar', value.dtype.str, value.tobytes())
    if isinstance(value, dict):
        return ('dict', tuple(sorted((key, _freeze_definition(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_freeze_definition(item) for item in value))
    if isinstance(value, float):
        return ('float', value.hex())
    if value is None or isinstance(value, (str, int, bool)):
        return (type(value).__name__, value)
    raise ValueError('Preserved sampling has an unsupported constraint definition')


class PreservedSurfaceProvenance:
    """Bind all frozen tools to exact inputs, including tools derived from CAD.

    The private CAD transfer imports each original input once and retains its
    shared references. Rebind those inputs after the checked transfer, never
    a rounded geometry hash. Exact numeric definitions and operand role/order
    bindings remain unchanged, and callers still rerun every mesh certificate.
    """

    _roles = ('solids', 'voids', 'hardware', 'source')

    @staticmethod
    def _inputs(payload):
        from .cad_features import (
            _coerce_to_solid, _payload_bounds, _payload_region_boxes, _payload_region_cylinders,
        )

        regions = payload.get('passive_regions') or {}
        shapes = []
        source = _coerce_to_solid(payload.get('source_cad_shape'))
        if source is not None:
            shapes.append(source)
        counts = [int(source is not None)]
        for key in ('explicit_solid_shapes', 'explicit_void_shapes'):
            selected = tuple(_coerce_to_solid(shape) for shape in regions.get(key) or ())
            if any(shape is None for shape in selected):
                raise ValueError('Preserved sampling requires original CAD selections')
            shapes.extend(selected)
            counts.append(len(selected))
        # Recovery materializes absent region keys as empty tuples, while the
        # exported solve payload can omit them. Snapshot the exact consumed
        # numeric records so those equivalent representations agree. Bounds
        # affect derived tools only when a fractional region is requested.
        fractional = {
            key: tuple((_payload_region_boxes if key.endswith('boxes') else _payload_region_cylinders)(payload, key))
            for key in ('solid_boxes', 'void_boxes', 'solid_cylinders', 'void_cylinders', 'joint_pin_cylinders')
        }
        definitions = {
            'source_and_selection_counts': counts,
            'bounds': _payload_bounds(payload) if any(fractional.values()) else None,
            'regions': fractional,
            'cad_features': tuple(payload.get('cad_features') or ()),
            'passive_cad_features': tuple(payload.get('passive_cad_features') or ()),
        }
        return tuple(shapes), _freeze_definition(definitions)

    def __init__(self, payload):
        self._shapes, self._definitions = self._inputs(payload)
        self._cad_fingerprints = tuple(cad_shape_fingerprint(shape) for shape in self._shapes)
        self._bindings = ()

    def bind_operands(self, cache):
        if self._bindings:
            raise ValueError('Preserved sampling operand provenance is already bound')
        self._bindings = tuple((role, tuple(cache[role])) for role in self._roles)

    def _validate_retained_state(self):
        if tuple(cad_shape_fingerprint(shape) for shape in self._shapes) != self._cad_fingerprints:
            raise ValueError('Preserved sampling input CAD state changed; regenerate the preserved surface')
        if tuple(role for role, _ in self._bindings) != self._roles:
            raise ValueError('Preserved sampling has incomplete operand provenance')
        for _, entries in self._bindings:
            for entry in entries:
                if not isinstance(entry, PreservedSurfaceOperand):
                    raise ValueError('Preserved sampling has an invalid bound operand')
                entry._validate_retained_state()

    def validate(self, payload, cache):
        self._validate_retained_state()
        shapes, definitions = self._inputs(payload)
        if (definitions != self._definitions
                or tuple(cad_shape_fingerprint(shape) for shape in shapes) != self._cad_fingerprints):
            raise ValueError('Preserved sampling does not match current CAD inputs or constraint definitions; '
                             'regenerate the preserved surface')
        for role, retained in self._bindings:
            current = cache.get(role)
            if (not isinstance(current, list) or len(current) != len(retained)
                    or any(first is not second for first, second in zip(current, retained))):
                raise ValueError('Preserved sampling operand role or order changed')

    def __getstate__(self):
        self._validate_retained_state()
        return {'version': 1, 'shapes': self._shapes, 'definitions': self._definitions,
                'bindings': self._bindings}

    def __setstate__(self, state):
        if state.get('version') != 1:
            raise ValueError('Unsupported preserved sampling provenance version')
        self._shapes, self._definitions = state['shapes'], state['definitions']
        self._bindings = state['bindings']
        self._cad_fingerprints = tuple(cad_shape_fingerprint(shape) for shape in self._shapes)
        self._validate_retained_state()
