# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Native CAD face and seam measurements for reconstruction review.

These measurements do not certify surface fairness. In particular, a large
normal jump can be an intended sharp feature. Keep face IDs and XYZ positions
so a reconstruction tool can distinguish those features from unwanted seams.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def inspect_cad_surfaces(shape, *, samples_per_edge=5):
    """Measure every face and shared edge without modifying the CAD.

    Normals are evaluated on the actual parametric surfaces, not on display
    triangles. Edge samples exclude endpoints, where CAD normals may be
    singular. The results are finite sampled diagnostics, not a global G1
    proof. Positions and areas use the input model's units (mm in PyLCSS).
    """
    if isinstance(samples_per_edge, bool) or int(samples_per_edge) != samples_per_edge or samples_per_edge < 1:
        raise ValueError('Seam inspection requires a positive integer sample count.')
    shape = shape.val() if callable(getattr(shape, 'val', None)) else shape
    faces = shape.Faces()
    records, edge_owners = [], defaultdict(list)
    for index, face in enumerate(faces):
        edges = face.Edges()
        area = float(face.Area())
        perimeter = float(sum(edge.Length() for edge in edges))
        box = face.BoundingBox()
        records.append({
            'face_id': index, 'surface_type': face.geomType(),
            'area_mm2': area, 'perimeter_mm': perimeter,
            'area_to_perimeter_squared': area / perimeter**2 if perimeter > 0 else None,
            'bounds_mm': [[box.xmin, box.ymin, box.zmin], [box.xmax, box.ymax, box.zmax]],
            'centre_mm': list(face.Center().toTuple()),
        })
        for edge in edges:
            # Hash buckets are verified with native topological identity below.
            edge_owners[edge.hashCode()].append((index, edge))

    seams, boundaries, nonmanifold = [], 0, 0
    parameters = np.arange(1, int(samples_per_edge) + 1) / (samples_per_edge + 1)
    for bucket in edge_owners.values():
        groups = []
        for index, edge in bucket:
            group = next((group for group in groups if edge.isSame(group[0][1])), None)
            if group is None:
                groups.append([(index, edge)])
            elif index not in {entry[0] for entry in group}:
                group.append((index, edge))
        for group in groups:
            # A periodic face can own both uses of its own seam. Such an edge
            # is not a cross-patch join and is not a free boundary of the solid.
            if len(group) == 1:
                boundaries += 1
                continue
            if len(group) != 2:
                nonmanifold += 1
                continue
            ids = [entry[0] for entry in group]
            edge = group[0][1]
            samples = []
            for parameter in parameters:
                point = edge.positionAt(float(parameter))
                entry = {'edge_fraction': float(parameter), 'xyz_mm': list(point.toTuple())}
                try:
                    normals = np.array([faces[index].normalAt(point).toTuple() for index in ids])
                    lengths = np.linalg.norm(normals, axis=1)
                    if not np.isfinite(normals).all() or np.any(lengths == 0):
                        raise ValueError('Undefined surface normal')
                    normals /= lengths[:, None]
                    entry.update(normal_vectors=normals.tolist(), normal_jump_degrees=float(
                        np.rad2deg(np.arccos(np.clip(normals[0] @ normals[1], -1., 1.)))))
                except Exception as error:
                    # Native kernels also raise their own exception classes
                    # at singularities. Retain the failed sample explicitly.
                    entry['normal_error'] = str(error)
                samples.append(entry)
            angles = [sample['normal_jump_degrees'] for sample in samples if 'normal_jump_degrees' in sample]
            seams.append({'seam_id': len(seams), 'face_ids': ids, 'surface_types': [records[index]['surface_type'] for index in ids],
                          'edge_length_mm': float(edge.Length()), 'samples': samples,
                          'maximum_sampled_normal_jump_degrees': max(angles) if angles else None})
    areas = [record['area_mm2'] for record in records]
    angles_by_pair = defaultdict(list)
    for seam in seams:
        if seam['maximum_sampled_normal_jump_degrees'] is not None:
            angles_by_pair[' / '.join(sorted(seam['surface_types']))].append(seam['maximum_sampled_normal_jump_degrees'])
    normal_summary = {
        pair: {'edge_count': len(values), 'percentiles': dict(zip(
            ('minimum', 'median', 'p90', 'p95', 'p99', 'maximum'),
            np.percentile(values, [0, 50, 90, 95, 99, 100]).tolist()))}
        for pair, values in sorted(angles_by_pair.items())
    }
    return {
        'method': 'Native CAD face dimensions and sampled parametric normals on shared edges',
        'geometry_changed': False, 'global_G1_continuity_certified': False,
        'sharp_feature_intent_classified': False,
        'face_id_convention': 'Zero-based order of input Shape.Faces()',
        'summary': {'face_count': len(faces), 'shared_edge_count': len(seams),
                    'single_face_edge_count': boundaries, 'nonmanifold_edge_count': nonmanifold,
                    'minimum_face_area_mm2': min(areas) if areas else None,
                    'face_area_percentiles_mm2': np.percentile(areas, [0, 5, 50, 95, 100]).tolist() if areas else [],
                    'normal_jump_degrees_by_surface_pair': normal_summary,
                    'smallest_face_ids': [face['face_id'] for face in sorted(records, key=lambda face: face['area_mm2'])[:10]],
                    'seams_with_undefined_normals': sum(any('normal_error' in sample for sample in seam['samples']) for seam in seams)},
        'faces': records, 'seams': seams,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cad_file', type=Path)
    parser.add_argument('output_json', type=Path)
    parser.add_argument('--samples-per-edge', type=int, default=5)
    args = parser.parse_args()
    import cadquery as cq
    if args.cad_file.suffix.lower() in {'.step', '.stp'}:
        shape = cq.importers.importStep(str(args.cad_file)).val()
    elif args.cad_file.suffix.lower() in {'.brep', '.brp'}:
        shape = cq.Shape.importBrep(str(args.cad_file))
    else:
        parser.error('Choose a STEP or BREP file.')
    report = inspect_cad_surfaces(shape, samples_per_edge=args.samples_per_edge)
    import hashlib
    report['source_file'] = str(args.cad_file.resolve())
    report['source_sha256'] = hashlib.sha256(args.cad_file.read_bytes()).hexdigest()
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
