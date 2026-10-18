# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Restore original prescribed geometry on recovered and exported surfaces.

The CAD inputs are tessellated at an explicit absolute deflection. The result
preserves those triangle solids, not an analytic CAD representation. Boolean
certificates are recomputed from independent double-precision mesh inputs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import trimesh


def _closed_mesh(vertices, faces, label: str, *, allow_degenerate: bool = False) -> trimesh.Trimesh:
    vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError(f'{label} has invalid triangle coordinates or indices')
    mesh = trimesh.Trimesh(vertices, np.asarray(faces, dtype=np.int64), process=False)
    if (not mesh.is_watertight or not mesh.is_winding_consistent
            or not np.isfinite(mesh.volume) or mesh.volume <= 0
            or (not allow_degenerate and np.any(mesh.area_faces <= 0))):
        raise ValueError(f'{label} must be a closed, consistently oriented positive-volume surface')
    return mesh


def _manifold(mesh: trimesh.Trimesh, label: str):
    from manifold3d import Error, Manifold, Mesh64

    result = Manifold(Mesh64(
        np.array(mesh.vertices, dtype=np.float64, order='C', copy=True),
        np.array(mesh.faces, dtype=np.uint64, order='C', copy=True),
    ))
    if result.status() != Error.NoError or result.is_empty():
        raise ValueError(f'{label} is not an oriented manifold ({result.status()})')
    return result


def _mesh(manifold, label: str, *, allow_degenerate: bool = False) -> trimesh.Trimesh:
    from manifold3d import Error

    if manifold.status() != Error.NoError or manifold.is_empty():
        raise RuntimeError(f'{label} did not retain material ({manifold.status()})')
    output = manifold.to_mesh64()
    return _closed_mesh(output.vert_properties[:, :3], output.tri_verts, label,
                        allow_degenerate=allow_degenerate)


def _boolean(parts, operation, label: str):
    from manifold3d import Error, Manifold

    result = Manifold.batch_boolean(parts, operation)
    if result.status() != Error.NoError:
        raise RuntimeError(f'{label} failed ({result.status()})')
    return result


def _partition_volume(shape, label: str, *, expected_empty: bool = False) -> float:
    from manifold3d import Error

    if shape.status() != Error.NoError:
        raise RuntimeError(f'{label} Boolean partition is invalid ({shape.status()})')
    if shape.is_empty():
        return 0.
    volume = float(shape.volume())
    if expected_empty:
        # Independent operands can leave zero-thickness coincident boundary
        # remnants with either volume sign at arithmetic roundoff. Measure
        # them conservatively: disconnected positive/negative material must
        # never cancel into an apparently empty partition.
        output = shape.to_mesh64()
        boundary = trimesh.Trimesh(output.vert_properties[:, :3], output.tri_verts, process=False)
        if (not np.isfinite(boundary.vertices).all() or not boundary.is_watertight
                or not boundary.is_winding_consistent or not np.isfinite(volume)):
            raise RuntimeError(f'{label} exterior partition is not a finite closed oriented boundary')
        component_volume = sum(abs(float(part.volume)) for part in boundary.split(only_watertight=False))
        conservative_volume = max(abs(volume), abs(float(boundary.volume)), component_volume)
        if not np.isfinite(conservative_volume):
            raise RuntimeError(f'{label} exterior partition has invalid volume')
        return conservative_volume
    # Independently verify the actual output boundary, not just backend status.
    # Coincident partition trims can contain zero-area bookkeeping triangles.
    # They add no volume; closure and orientation remain mandatory. Delivered
    # surfaces and operands still prohibit all zero-area triangles.
    mesh = _mesh(shape, label, allow_degenerate=True)
    if not np.isfinite(volume) or volume <= 0:
        raise RuntimeError(f'{label} Boolean partition has invalid volume')
    if abs(float(mesh.volume) - volume) > volume * 1e-9:
        raise RuntimeError(f'{label} Boolean and triangle volumes disagree')
    return volume


def _assert_contained(material: trimesh.Trimesh, domain: trimesh.Trimesh, label: str) -> dict:
    """Require an empty exterior AND a complete independent interior partition."""
    from manifold3d import OpType

    reference = abs(float(material.volume))
    # A model-unit floor can exceed an entire small prescribed region.
    tolerance = reference * 1e-9
    def partitions(required, surrounding):
        # Construct separate operands for both checks; do not reuse Boolean history.
        outside = _boolean([_manifold(required, label), _manifold(surrounding, label)],
                           OpType.Subtract, label)
        inside = _boolean([_manifold(required, label), _manifold(surrounding, label)],
                          OpType.Intersect, label)
        return (_partition_volume(outside, f'{label} exterior', expected_empty=True),
                _partition_volume(inside, f'{label} interior'))

    outside_volume, inside_volume = partitions(material, domain)
    retry = None
    if outside_volume <= tolerance and inside_volume > reference + tolerance:
        # Intersection cannot add material. Nearly coincident cuts can produce
        # that impossible result when evaluated far from their local origin.
        # A rigid translation of BOTH unchanged operands is a representation
        # retry, not extra volume allowance. Recheck both independent partitions
        # against the original required volume and unchanged relative tolerance.
        origin = material.bounds.mean(axis=0)
        local_material = _closed_mesh(material.vertices-origin, material.faces, label)
        local_domain = _closed_mesh(domain.vertices-origin, domain.faces, label)
        if (abs(float(local_material.volume)-reference) > tolerance
                or abs(float(local_domain.volume)-float(domain.volume)) > abs(float(domain.volume))*1e-9):
            raise RuntimeError(f'{label} local coordinate translation changed operand volume')
        retry = {'reason': 'Intersection volume exceeded required material volume',
                 'original_outside_volume': outside_volume, 'original_inside_volume': inside_volume,
                 'translation': (-origin).tolist(), 'volume_tolerance': tolerance}
        outside_volume, inside_volume = partitions(local_material, local_domain)
    if outside_volume > tolerance or abs(inside_volume - reference) > tolerance:
        raise RuntimeError(f'{label} was not preserved: exterior volume {outside_volume:.17g}, '
                           f'interior volume {inside_volume:.17g}, required {reference:.17g}; '
                           f'volume tolerance {tolerance:.17g}')
    result = {'required_volume': reference, 'outside_volume': outside_volume,
              'inside_volume': inside_volume, 'volume_tolerance': tolerance}
    if retry is not None:
        result['local_frame_retry'] = retry
    return result


def _assert_excluded(material: trimesh.Trimesh, void: trimesh.Trimesh, label: str) -> dict:
    from manifold3d import OpType

    reference = abs(float(material.volume))
    tolerance = min(reference, abs(float(void.volume))) * 1e-9
    intrusion = _boolean([_manifold(material, label), _manifold(void, label)],
                         OpType.Intersect, label)
    retained = _boolean([_manifold(material, label), _manifold(void, label)],
                        OpType.Subtract, label)
    intrusion_volume = _partition_volume(intrusion, f'{label} intrusion', expected_empty=True)
    retained_volume = _partition_volume(retained, f'{label} retained material')
    if intrusion_volume > tolerance or abs(retained_volume - reference) > tolerance:
        raise RuntimeError(f'{label} contains forbidden material: intrusion volume {intrusion_volume:.9g}')
    return {'intrusion_volume': intrusion_volume, 'retained_volume': retained_volume,
            'material_volume': reference, 'volume_tolerance': tolerance}


def _region_payload(payload: dict) -> dict:
    """Match recovery's persisted-region locations without dropping bad records."""
    from . import cad_features as cf

    normalized = dict(payload)
    regions = dict(payload.get('passive_regions') or {})
    for key in ('solid_boxes', 'void_boxes', 'solid_cylinders', 'void_cylinders',
                'joint_pin_cylinders', 'explicit_solid_shapes', 'explicit_void_shapes'):
        if key not in regions and payload.get(key) is not None:
            regions[key] = payload[key]
    recovered = payload.get('recovered_shape')
    if isinstance(recovered, dict):
        # A display mesh copy must never replace the original CAD selection.
        for key, value in (recovered.get('preserved_lattice_geometry') or {}).items():
            regions.setdefault(key, value)
    normalized['passive_regions'] = regions
    for key in ('explicit_solid_shapes', 'explicit_void_shapes'):
        for raw in regions.get(key) or ():
            shape = cf._coerce_to_solid(raw)
            if shape is None or not shape.isValid() or not shape.Solids():
                raise ValueError(f'Invalid original prescribed CAD in {key}; '
                                 'a saved display mesh cannot replace the original CAD selection')
    bounds = cf._payload_bounds(normalized)
    for key in ('solid_boxes', 'void_boxes', 'solid_cylinders', 'void_cylinders', 'joint_pin_cylinders'):
        records = regions.get(key) or ()
        if not records:
            continue
        if bounds is None:
            raise ValueError('Prescribed fractional geometry requires physical study bounds')
        boxes = key.endswith('boxes')
        parsed = (cf._payload_region_boxes if boxes else cf._payload_region_cylinders)(normalized, key)
        constructor = cf._region_box_to_solid if boxes else cf._region_cylinder_to_solid
        if len(parsed) != len(records) or any(constructor(item, bounds) is None for item in parsed):
            raise ValueError(f'Invalid prescribed geometry in {key}')
    for role in ('solid', 'void'):
        for kind in ('box', 'cylinder'):
            constructor = cf._physical_box_to_solid if kind == 'box' else cf._physical_cylinder_to_solid
            for record in cf._payload_physical_cad_features(normalized, role, kind):
                if constructor(record) is None:
                    raise ValueError(f'Invalid physical prescribed {role} {kind}')
    return normalized


def restore_preserved_surface(vertices, faces, payload: dict[str, Any], *,
                              tessellation_tolerance: float | None = None,
                              validate_only: bool = False,
                              sampling_cache: dict[str, Any] | None = None):
    """Return certified restored triangles and their tessellated-CAD provenance.

    Add every requested material layer, intersect the original source boundary,
    and remove every protected void. Hardware remains separate assembly solids.
    No smoothing or mesh reduction may follow this final geometry operation.
    validate_only checks every certificate against the supplied boundary and
    returns its original coordinates and topology without another restoration.
    A sampling cache retains the original chordal representation across CAD
    serialization. Its CAD identities and mesh hashes are checked on every use;
    it never replaces the independent material and intersection certificates.
    """
    from manifold3d import Manifold, OpType
    from .cad_features import (
        _coerce_to_solid, _collect_passive_tools, _assert_source_contains_material,
        _normalize_boolean_trims,
    )
    from .geometry_lattice import geometry_triangle_mesh
    from .mesh_screening import screen_mesh_intersections
    from .preserved_surface_sampling import PreservedSurfaceOperand, PreservedSurfaceProvenance
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy

    if not isinstance(payload, dict):
        raise TypeError('Surface preservation requires a topology result dictionary')
    original = _closed_mesh(vertices, faces, 'Recovered surface')
    unchanged_input = ((original.vertices.copy(), original.faces.copy()) if validate_only else None)
    input_screening = screen_mesh_intersections(original.vertices, original.faces,
                                                include_boundary_contacts=False)
    if (input_screening.get('screening_scope') != 'full'
            or input_screening.get('self_intersecting_faces') != 0):
        raise ValueError('Recovered surface has unresolved triangle intersections')
    _manifold(original, 'Recovered surface')
    normalized = _region_payload(payload)
    solids, cuts, hardware = _collect_passive_tools(
        normalized, void_axial_margin=.02, void_radial_margin=0.)
    raw_source = payload.get('source_cad_shape')
    source = _coerce_to_solid(raw_source)
    if raw_source is not None and (source is None or not source.isValid() or not source.Solids()):
        raise ValueError('Original source CAD is not a valid solid')
    scale = float(np.max(original.extents))
    if source is not None:
        box = source.BoundingBox()
        scale = max(scale, box.xlen, box.ylen, box.zlen)
    if tessellation_tolerance is None and isinstance(sampling_cache, dict):
        # Clipping can reduce the supplied surface extent. Revalidation must
        # keep the recorded absolute deflection of its original operands.
        tessellation_tolerance = sampling_cache.get('tessellation_tolerance')
    tolerance = (min(.01, max(1e-6, scale * 1e-5))
                 if tessellation_tolerance is None else float(tessellation_tolerance))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('CAD tessellation tolerance must be finite and positive')
    original_solids = [solid for tool in solids for solid in tool.Solids()]
    original_voids = [solid for tool in cuts for solid in tool.Solids()]
    original_hardware = [solid for tool in hardware for solid in tool.Solids()]
    operands = {'solids': original_solids, 'voids': original_voids,
                'hardware': original_hardware, 'source': [] if source is None else [source]}
    frozen = {'revision': 2, 'tessellation_tolerance': tolerance,
              'tessellation_angular_tolerance': .15}
    bind_provenance = sampling_cache is None
    if sampling_cache is not None:
        if (not isinstance(sampling_cache, dict)
                or sampling_cache.get('revision') not in (1, 2)
                or any(sampling_cache.get(key) != value for key, value in frozen.items() if key != 'revision')
                or any(not isinstance(sampling_cache.get(key), list)
                       or len(sampling_cache[key]) != len(items) for key, items in operands.items())):
            raise ValueError('Preserved surface sampling cache does not match current CAD constraints or tolerance')
        if sampling_cache['revision'] == 1:
            # Upgrade older retained samples only after their original strict
            # raw-CAD checks succeed for every current derived tool. A v1
            # angled bore whose imported coefficients differ still requires
            # regeneration; no rounded identity is introduced for migration.
            for role, items in operands.items():
                for shape, entry in zip(items, sampling_cache[role]):
                    if not isinstance(entry, PreservedSurfaceOperand):
                        raise ValueError('Preserved surface sampling cache contains an invalid operand')
                    entry.validated_mesh(shape)
            provenance = PreservedSurfaceProvenance(normalized)
            bind_provenance = True
        else:
            # Check the actual inputs, not newly regenerated derived bore BREP:
            # importing an unchanged direction can renormalize its last bits.
            provenance = sampling_cache.get('input_provenance')
            if not isinstance(provenance, PreservedSurfaceProvenance):
                raise ValueError('Preserved surface sampling cache has no input provenance')
            provenance.validate(normalized, sampling_cache)
    else:
        provenance = PreservedSurfaceProvenance(normalized)
    report = {'applied': bool(source is not None or solids or cuts or hardware),
              'certified': False, 'source_cad_applied': source is not None,
              'preserved_solid_count': len(solids), 'preserved_void_count': len(cuts),
              'hardware_count': len(hardware), 'tessellation_tolerance': tolerance,
              'tessellation_angular_tolerance': .15,
              'geometry_representation': 'Original CAD tessellated at absolute model-unit deflection; not analytic CAD',
              'method': 'Double-precision Manifold mesh union, source intersection and protected-void difference',
              'certificate_method': 'Independent paired mesh Boolean partitions with triangle-volume verification',
              'input_intersections': input_screening,
              'post_restoration_smoothing': False, 'post_restoration_reduction': False}
    report['validation_only'] = bool(validate_only)
    report['sampling_cache_reused'] = sampling_cache is not None
    if validate_only:
        report['method'] = 'Independent preservation certificates on the unchanged supplied surface'
    if not report['applied']:
        return original.vertices.copy(), original.faces.copy(), {**report, 'certified': True}

    def isolated(shape):
        # Native meshing may populate pcurves as well as triangle caches. Work
        # on private topology and geometry so the original CAD remains intact.
        return cq.Shape.cast(BRepBuilderAPI_Copy(shape.wrapped, True, False).Shape())

    def sampled(shape, label, role, index):
        if sampling_cache is None:
            try:
                mesh = geometry_triangle_mesh(isolated(shape), tolerance)
            except ValueError as error:
                raise ValueError(f'{label} could not be tessellated as a closed material boundary: {error}') from error
            entry = PreservedSurfaceOperand(shape, mesh.vertices, mesh.faces)
        else:
            entry = sampling_cache[role][index]
            points, triangles = entry.validated_mesh()
            mesh = trimesh.Trimesh(points, triangles, process=False)
        mesh = _closed_mesh(mesh.vertices, mesh.faces, label)
        screening = screen_mesh_intersections(mesh.vertices, mesh.faces, include_boundary_contacts=False)
        if screening.get('screening_scope') != 'full' or screening.get('self_intersecting_faces') != 0:
            raise ValueError(f'{label} has unresolved triangle intersections')
        _manifold(mesh, label)
        frozen.setdefault(role, []).append(entry)
        return mesh

    # Keep individual solid identities: a compound must not let one absent
    # thin layer hide beneath a larger aggregate volume tolerance.
    report['original_cad_layer_source_containment'] = []
    if source is not None:
        for index, solid in enumerate(original_solids):
            # Independent curved tessellations have different chords. Prove
            # the actual CAD selection is contained before permitting its
            # sampled representation to be clipped to the sampled source.
            representation = 'original'
            try:
                _assert_source_contains_material(isolated(solid), isolated(source))
            except (ValueError, RuntimeError) as error:
                # OCC can misclassify an entire coincident cylindrical layer
                # after its triangulation caches have been populated. A STEP
                # trim normalization may repair that bookkeeping only after
                # checking identical surfaces, solid count, volume and bounds.
                # Keep sampling the original layer; the retry changes only
                # the representation used for this independent CAD proof.
                try:
                    canonical = _normalize_boolean_trims(isolated(solid))
                    _assert_source_contains_material(canonical, isolated(source))
                except Exception as retry_error:
                    raise RuntimeError(f'Prescribed layer {index} was not preserved: '
                                       f'original CAD is not contained in its source: {error}; '
                                       f'exact trim retry failed: {retry_error}') from retry_error
                representation = 'exact_normalized_trims'
            report['original_cad_layer_source_containment'].append(
                {'layer': index, 'verified': True, 'verification_representation': representation})
    solid_meshes = [sampled(solid, f'Prescribed layer {index}', 'solids', index)
                   for index, solid in enumerate(original_solids)]
    void_meshes = [sampled(solid, f'Protected void {index}', 'voids', index)
                  for index, solid in enumerate(original_voids)]
    hardware_meshes = [sampled(solid, f'Assembly hardware {index}', 'hardware', index)
                      for index, solid in enumerate(original_hardware)]
    source_mesh = sampled(source, 'Original source boundary', 'source', 0) if source is not None else None
    for key in operands:
        frozen.setdefault(key, [])
    if bind_provenance:
        provenance.bind_operands(frozen)
    frozen['input_provenance'] = provenance
    provenance.validate(normalized, frozen)
    if validate_only:
        material = original
        if hardware_meshes:
            # Separate only a proof operand; the delivered assembly is never
            # rewritten. The independent coverage certificate below verifies
            # that no source/void violation was hidden by this partition.
            body = _boolean([_manifold(original, 'Supplied assembly'),
                             *[_manifold(mesh, 'Assembly hardware') for mesh in hardware_meshes]],
                            OpType.Subtract, 'Assembly material proof partition')
            material = _mesh(body, 'Assembly material proof partition')
    else:
        body = _boolean([_manifold(original, 'Recovered surface'),
                         *[_manifold(mesh, 'Prescribed layer') for mesh in solid_meshes]],
                        OpType.Add, 'Prescribed material union')
        if source_mesh is not None:
            body = _boolean([body, _manifold(source_mesh, 'Original source boundary')],
                            OpType.Intersect, 'Original source intersection')
        if void_meshes:
            body = _boolean([body, *[_manifold(mesh, 'Protected void') for mesh in void_meshes]],
                            OpType.Subtract, 'Protected void subtraction')
        material = _mesh(body, 'Restored material surface')
    if not validate_only:
        retained_original = _manifold(original, 'Original recovered material')
        if source_mesh is not None:
            retained_original = _boolean([retained_original, _manifold(source_mesh, 'Original source boundary')],
                                         OpType.Intersect, 'Expected original source clipping')
        if void_meshes:
            retained_original = _boolean([retained_original, *[_manifold(void, 'Protected void') for void in void_meshes]],
                                         OpType.Subtract, 'Expected original void exclusion')
        report['original_material_certificate'] = (
            {'fully_excluded_by_source_or_voids': True} if retained_original.is_empty() else
            _assert_contained(_mesh(retained_original, 'Expected original material'),
                              material, 'Original recovered material'))
    certificates = []
    for index, mesh in enumerate(solid_meshes):
        expected = _manifold(mesh, f'Prescribed layer {index}')
        if void_meshes:
            expected = _boolean([expected, *[_manifold(void, 'Protected void') for void in void_meshes]],
                                OpType.Subtract, f'Prescribed layer {index} void exclusion')
        if expected.is_empty():
            certificates.append({'fully_excluded_by_prescribed_void': True})
            continue
        sampled_required_volume = _partition_volume(expected, f'Prescribed layer {index} before source clipping')
        if source_mesh is not None:
            expected = _boolean([expected, _manifold(source_mesh, 'Original source boundary')],
                                OpType.Intersect, f'Prescribed layer {index} sampled source clipping')
            if expected.is_empty():
                raise RuntimeError(f'Prescribed layer {index} was erased by source tessellation')
        expected_mesh = _mesh(expected, 'Expected prescribed material')
        certificate = _assert_contained(expected_mesh, material, f'Prescribed layer {index}')
        certificate.update(sampled_required_volume_before_source_clip=sampled_required_volume,
                           sampled_source_clip_volume_delta=sampled_required_volume-float(expected_mesh.volume),
                           original_cad_source_containment_verified=source is not None)
        certificates.append(certificate)
    report['material_certificates'] = certificates
    report['source_certificate'] = (_assert_contained(material, source_mesh, 'Original source boundary')
                                    if source_mesh is not None else None)
    report['void_certificates'] = [_assert_excluded(material, void, f'Protected void {index}')
                                   for index, void in enumerate(void_meshes)]
    if hardware_meshes:
        # Compose keeps the pin/body assembly relation; union would bond them.
        body = Manifold.compose([_manifold(material, 'Restored material'),
                                 *[_manifold(mesh, 'Assembly hardware') for mesh in hardware_meshes]])
    if validate_only:
        if hardware_meshes:
            report['assembly_coverage_certificate'] = _assert_contained(
                original, _mesh(body, 'Expected material and hardware assembly'), 'Supplied assembly')
        result = original
    else:
        result = _mesh(body, 'Final preserved surface')
    report['hardware_certificates'] = [_assert_contained(mesh, result, f'Assembly hardware {index}')
                                       for index, mesh in enumerate(hardware_meshes)]
    screening = screen_mesh_intersections(result.vertices, result.faces, include_boundary_contacts=False)
    if screening.get('screening_scope') != 'full' or screening.get('self_intersecting_faces') != 0:
        raise RuntimeError('Preserved surface has unresolved triangle intersections')
    if validate_only:
        # The delivered surface is exactly the supplied material. Retention is
        # proved by coordinate/topology identity, with zero numerical allowance;
        # source, void, layer and hardware checks above remain independent.
        if (not np.array_equal(result.vertices, unchanged_input[0])
                or not np.array_equal(result.faces, unchanged_input[1])):
            raise RuntimeError('Preservation validation changed the supplied surface')
        report['original_material_certificate'] = {
            'method': 'Unchanged supplied coordinates and triangle indices',
            'coordinate_identity_verified': True, 'topology_identity_verified': True,
            'required_volume': float(original.volume), 'inside_volume': float(result.volume),
            'outside_volume': 0., 'volume_tolerance': 0.,
        }
    report.update(certified=True, material_solid_count=len(solid_meshes),
                  input_volume=float(original.volume), material_volume=float(material.volume),
                  output_volume=float(result.volume), output_faces=len(result.faces),
                  is_watertight=True, winding_consistent=True, intersections=screening)
    # Raw arrays and CAD objects belong in the private recovered-shape payload,
    # not in the public JSON certificate/report.
    report['_sampling_cache'] = frozen
    return np.asarray(result.vertices), np.asarray(result.faces), report
