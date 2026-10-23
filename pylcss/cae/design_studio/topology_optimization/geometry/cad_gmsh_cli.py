# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""CAD meshing through an optional, separately installed Gmsh executable.

Only STEP, GEO and MSH files cross the process boundary. Gmsh's Python API or
shared libraries are never imported. The CAD is neither healed nor replaced.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from pylcss.cae.solver_backends.execution import resolve_executable, run_process

logger = logging.getLogger(__name__)

# Gmsh's 0.1 degree default rejects valid, very sharp material wedges as
# overlapping facets. Keep a positive guard, and independently screen the
# CAD and native triangles instead of treating this angular heuristic as an
# intersection test. This changes no coordinates or completion tolerances.
_FACET_OVERLAP_ANGLE_DEGREES = .01


def resolve_gmsh_executable(explicit=None):
    """Use the same executable configuration and discovery as external solvers."""
    root = Path(__file__).resolve().parents[5] / 'external_solvers' / 'gmsh'
    candidates = [root / 'gmsh.exe', root / 'gmsh']
    if root.is_dir():
        candidates.extend(sorted(root.glob('*/gmsh.exe')))
        candidates.extend(sorted(root.glob('*/bin/gmsh')))
    return resolve_executable(explicit, ('PYLCSS_GMSH_EXE',), (*candidates, 'gmsh'))


def _geo_script(maximum, minimum, volume_maximum, timeout_s, grading=.3):
    """Fixed local filenames avoid injecting arbitrary paths into GEO syntax."""
    settings = {
        'General.AbortOnError': 4, 'General.NoPopup': 1,
        'General.NumThreads': 4, 'General.Terminal': 1,
        'Geometry.OCCAutoFix': 0, 'Geometry.OCCFixDegenerated': 0,
        'Geometry.OCCFixSmallEdges': 0, 'Geometry.OCCFixSmallFaces': 0,
        'Geometry.OCCSewFaces': 0, 'Geometry.OCCMakeSolids': 0,
        'Geometry.OCCAutoEmbed': 0,
        'Mesh.MaxNumThreads1D': 4, 'Mesh.MaxNumThreads2D': 4,
        'Mesh.MaxNumThreads3D': 4, 'Mesh.Algorithm': 6,
        'Mesh.Algorithm3D': 1, 'Mesh.OptimizeNetgen': 0,
        'Mesh.AngleToleranceFacetOverlap': _FACET_OVERLAP_ANGLE_DEGREES,
        'Mesh.MeshSizeFromCurvature': 0,
        'Mesh.MeshSizeMin': minimum, 'Mesh.MeshSizeMax': maximum,
        'Mesh.MeshSizeExtendFromBoundary': 1,
        'Mesh.MeshOnlyEmpty': 1, 'Mesh.ElementOrder': 1,
        'Mesh.MshFileVersion': 2.2, 'Mesh.Binary': 0,
        'Mesh.SaveAll': 1, 'Mesh.Renumber': 0,
    }
    lines = [f'{key} = {value:.17g};' for key, value in settings.items()]
    lines.extend(('SetFactory("OpenCASCADE");', 'v() = ShapeFromFile("cad_input.step");',
                  'SyncModel;'))
    # Reloading a native mesh does not restore all Gmsh meshing state. Keep this
    # ordinary external process alive while Python validates the saved surface.
    # This bounded file handshake imports no API and never invokes another fit.
    lines.extend(('Mesh 2;', 'Save "surface.msh";',
                  'Printf("ready") > "surface.ready";',
                  f'For _pylcss_gate In {{0:{math.ceil(timeout_s / .05)}}}',
                  '  If (!FileExists("surface.accepted"))',
                  '    Sleep 0.05;', '  EndIf', 'EndFor',
                  'If (!FileExists("surface.accepted"))',
                  '  Error("CAD surface was not accepted before timeout");',
                  '  Exit 1;', 'EndIf',
                  f'Mesh.MeshSizeMax = {volume_maximum:.17g};'))
    if maximum < volume_maximum:
        # A physical size slope, rather than an invented equivalence with
        # Netgen's algorithm-specific grading. This affects only volume sizing.
        distance = (volume_maximum - maximum) / grading
        lines.extend(('_pylcss_surfaces[] = Surface{:};', 'Field[1] = Distance;',
                      'Field[1].SurfacesList = {_pylcss_surfaces[]};',
                      'Field[1].Sampling = 20;',
                      'Field[2] = Threshold;', 'Field[2].InField = 1;',
                      f'Field[2].SizeMin = {maximum:.17g};',
                      f'Field[2].SizeMax = {volume_maximum:.17g};',
                      'Field[2].DistMin = 0;',
                      f'Field[2].DistMax = {distance:.17g};',
                      'Field[2].Sigmoid = 0;', 'Background Field = 2;',
                      'Mesh.MeshSizeExtendFromBoundary = 0;',
                      'Mesh.MeshSizeFromPoints = 0;'))
    lines.extend(('Mesh 3;', 'Save "volume.msh";'))
    return '\n'.join(lines) + '\n'


def _mesh_arrays(path, *, volume=False):
    import meshio

    data = meshio.read(path)
    points = np.asarray(data.points[:, :3], dtype=float)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Gmsh returned missing or nonfinite mesh coordinates.')
    allowed = {'vertex', 'line', 'triangle', 'tetra'}
    if any(block.type not in allowed for block in data.cells):
        raise ValueError('Gmsh returned unsupported or nonlinear mesh elements.')
    triangles = [block.data for block in data.cells if block.type == 'triangle']
    tetrahedra = [block.data for block in data.cells if block.type == 'tetra']
    if not triangles:
        raise ValueError('Gmsh returned no native CAD surface triangles.')
    triangles = np.concatenate(triangles).astype(np.int64, copy=False)
    tetrahedra = (np.concatenate(tetrahedra).astype(np.int64, copy=False)
                 if tetrahedra else np.empty((0, 4), dtype=np.int64))
    if volume and not len(tetrahedra):
        raise ValueError('Gmsh returned no tetrahedral material.')
    if not volume and len(tetrahedra):
        raise ValueError('The Gmsh surface stage unexpectedly generated volume elements.')
    for array in (triangles, tetrahedra):
        if array.size and (array.min() < 0 or array.max() >= len(points)):
            raise ValueError('Gmsh mesh connectivity references an unavailable node.')
    return points, triangles, tetrahedra


def _closed_boundary(points, triangles):
    import trimesh

    if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
        raise ValueError('Gmsh native material boundary contains duplicate facets.')
    boundary = trimesh.Trimesh(points, triangles, process=False)
    if not boundary.is_watertight:
        raise ValueError('Gmsh native material boundary is not closed.')
    if not boundary.is_winding_consistent:
        raise ValueError('Gmsh native material boundary has inconsistent winding.')
    volume = float(boundary.volume)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError('Gmsh native boundary has nonpositive oriented material volume.')
    return {'triangles': len(triangles), 'closed': True,
            'winding_consistent': True, 'oriented_volume_mm3': volume}


def _screen_boundary(points, triangles):
    """Reject real crossings independently of the mesher's wedge-angle guard."""
    from .mesh_screening import screen_mesh_intersections

    report = screen_mesh_intersections(points, triangles)
    if report.get('screening_scope') != 'full' or report.get('self_intersecting_faces') != 0:
        raise ValueError('Gmsh native CAD boundary intersects itself or could not be fully screened.')
    return report


def _same_surface(before_points, before_faces, after_points, after_faces):
    """Require unchanged XYZ triples and the exact same triangle connectivity."""
    from scipy.spatial import cKDTree

    before_used = np.unique(before_faces)
    after_used = np.unique(after_faces)
    if len(before_used) != len(after_used) or len(before_faces) != len(after_faces):
        raise ValueError('Gmsh changed the accepted native CAD surface during volume generation.')
    distances, local = cKDTree(after_points[after_used]).query(before_points[before_used])
    # Both files serialize the same doubles. Permit only their decimal-format
    # roundoff, never a geometric tolerance or vertex projection.
    scale = max(float(np.max(np.abs(before_points[before_used]))), 1.)
    bound = np.finfo(float).eps * scale * 64
    if np.max(distances) > bound or len(np.unique(local)) != len(local):
        raise ValueError('Gmsh moved or merged accepted CAD boundary vertices.')
    mapping = np.full(len(before_points), -1, dtype=np.int64)
    mapping[before_used] = after_used[local]
    first = np.sort(mapping[before_faces], axis=1)
    second = np.sort(after_faces, axis=1)
    first = first[np.lexsort(first.T[::-1])]
    second = second[np.lexsort(second.T[::-1])]
    if not np.array_equal(first, second):
        raise ValueError('Gmsh remeshed the accepted native CAD surface.')
    return {'native_surface_facets_retained': True,
            'maximum_surface_coordinate_roundoff_mm': float(np.max(distances)),
            'surface_coordinate_roundoff_allowance_mm': bound}


def _tetrahedral_boundary(points, tetrahedra, native_faces):
    """Verify the material actually filled, rather than a retained surface list."""
    if len(np.unique(np.sort(tetrahedra, axis=1), axis=0)) != len(tetrahedra):
        raise ValueError('Gmsh returned duplicate material tetrahedra.')
    corners = points[tetrahedra]
    triple = np.einsum('ij,ij->i', corners[:, 1] - corners[:, 0],
                      np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]))
    if not np.isfinite(triple).all() or np.any(triple <= 0):
        raise ValueError('Gmsh returned inverted or collapsed tetrahedra.')
    faces = np.concatenate([tetrahedra[:, order] for order in
                            ([0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3])])
    keys = np.sort(faces, axis=1)
    _, first, inverse, counts = np.unique(keys, axis=0, return_index=True,
                                         return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Gmsh tetrahedra have nonmanifold shared facets.')
    inversions = ((faces[:, 0] > faces[:, 1]).astype(int)
                  + (faces[:, 0] > faces[:, 2]) + (faces[:, 1] > faces[:, 2]))
    orientation_sum = np.bincount(inverse, weights=1 - 2 * (inversions % 2))
    if np.any(orientation_sum[counts == 2] != 0):
        raise ValueError('Gmsh material tetrahedra have incompatible internal facet orientations.')
    external = faces[first[counts == 1]]
    _closed_boundary(points, external)
    native = np.sort(native_faces, axis=1)
    actual = np.sort(external, axis=1)
    native = native[np.lexsort(native.T[::-1])]
    actual = actual[np.lexsort(actual.T[::-1])]
    if not np.array_equal(native, actual):
        raise ValueError('Actual tetrahedral exterior differs from the accepted native CAD surface.')
    return float(triple.sum() / 6.), {'positive_tetrahedra': len(tetrahedra),
            'opposite_internal_facet_orientations': True,
            'actual_tetrahedral_exterior_closed': True,
            'actual_tetrahedral_exterior_winding_consistent': True,
            'actual_tetrahedral_exterior_equals_native_CAD_surface': True}


def cad_tet_mesh_cli(shape, element_size, directory, *, name='analysis',
                     minimum_size=0., grading=.3, element_order='C3D10',
                     max_elements=500000, reference_volume=None, executable=None,
                     timeout_s=300., cancel_callback=None):
    """Run one native surface stage and one volume stage, with no retry loop."""
    from skfem import MeshTet, MeshTet2
    from pylcss.cae.design_studio.fem.quality import attach_mesh_quality
    from ..verification._common import finite_nonnegative, validation_integer, validated_cad_shape
    from .amrto_volume import measure_cad_volume

    maximum = finite_nonnegative(element_size, name='CAD maximum element size', positive=True)
    minimum = finite_nonnegative(minimum_size, name='CAD minimum element size')
    timeout = finite_nonnegative(timeout_s, name='CAD mesher timeout', positive=True)
    grading = finite_nonnegative(grading, name='CAD mesh grading', positive=True)
    if minimum > maximum:
        raise ValueError('CAD minimum element size cannot exceed its maximum.')
    if grading > 1 or element_order not in ('C3D4', 'C3D10'):
        raise ValueError('Invalid CAD mesh grading or tetrahedral element order.')
    maximum_elements = validation_integer(max_elements, name='CAD maximum elements')
    if not name or Path(name).name != name or any(char in name for char in ('/', '\\', '\0')):
        raise ValueError('CAD mesh job name must be a plain filename.')
    program = resolve_gmsh_executable(executable)
    if program is None:
        raise ValueError('Standalone Gmsh executable was not found. Configure PYLCSS_GMSH_EXE or install the external CAD mesher.')
    shape = validated_cad_shape(shape)
    if reference_volume is None:
        reference_volume, _ = measure_cad_volume(shape)
    reference = finite_nonnegative(reference_volume, name='CAD reference material volume', positive=True)
    folder = Path(directory).resolve() / f'{name}_gmsh'
    folder.mkdir(parents=True, exist_ok=True)
    boundary_maximum = max(minimum, maximum / 3.)
    started = time.monotonic()
    report = {'backend': 'Gmsh standalone CLI', 'executable': program,
              'requested_maximum_size_mm': maximum, 'requested_minimum_size_mm': minimum,
              'effective_boundary_maximum_size_mm': boundary_maximum,
              'effective_volume_maximum_size_mm': maximum,
              'effective_minimum_size_mm': minimum, 'requested_grading': grading,
              'grading_policy': ('Linear volume-size slope from sampled CAD surface distance; native Netgen equivalence is not assumed'
                                 if boundary_maximum < maximum else 'Uniform size: requested minimum equals maximum'),
              'background_size_field_applied': boundary_maximum < maximum,
              'volume_size_gradient': grading if boundary_maximum < maximum else 0.,
              'volume_size_transition_distance_mm': ((maximum - boundary_maximum) / grading
                                                     if boundary_maximum < maximum else 0.),
              'surface_distance_sampling': 20,
              'reference_cad_volume_mm3': reference, 'geometry_healing_enabled': False,
              'python_gmsh_api_used': False, 'maximum_elements': maximum_elements,
              'element_order': element_order, 'surface_stages': 0, 'volume_stages': 0,
              'facet_overlap_angle_degrees': _FACET_OVERLAP_ANGLE_DEGREES,
              'retry_count': 0, 'passed': False}
    report_path = folder / 'meshing_report.json'

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    try:
        from .occ_shapes import _assess_brep_self_intersections
        report['cad_self_intersections'] = _assess_brep_self_intersections(shape)
        if not report['cad_self_intersections']['passed']:
            raise ValueError('CAD contains geometric self-intersections; rebuild the CAD before FEA.')
        source = folder / 'cad_input.step'
        shape.exportStep(str(source))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        report['exported_cad_step_sha256'] = source_hash
        for filename in ('surface.msh', 'surface.ready', 'surface.accepted', 'volume.msh'):
            stale = folder / filename
            if stale.exists():
                stale.unlink()
        script = folder / 'cad_mesh.geo'
        script.write_text(_geo_script(boundary_maximum, minimum, maximum, timeout, grading), encoding='utf-8')
        surface_state = {}

        def check_surface_and_cancel():
            if cancel_callback is not None and cancel_callback():
                return True
            if not surface_state and (folder / 'surface.ready').is_file():
                before_points, before_faces, _ = _mesh_arrays(folder / 'surface.msh')
                report['native_surface'] = _closed_boundary(before_points, before_faces)
                report['native_surface_intersections'] = _screen_boundary(before_points, before_faces)
                surface_state.update(points=before_points, faces=before_faces)
                if cancel_callback is not None and cancel_callback():
                    return True
                report['volume_stages'] = 1; save()
                (folder / 'surface.accepted').write_text('accepted\n', encoding='ascii')
            return False

        report['surface_stages'] = 1; save()
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise ValueError('CAD mesher exceeded its timeout before launch.')
        process = run_process([program, script.name, '-parse_and_exit', '-nopopup', '-v', '4'],
                              cwd=folder, timeout_s=remaining, stdout_file=folder / 'native.log',
                              cancel_callback=check_surface_and_cancel)
        native_log = (folder / 'native.log').read_text(encoding='utf-8', errors='replace')
        # A successful exit can still contain a useful element-shape warning.
        # Keep it with the analysis instead of losing it with the temp files.
        report['native_warnings'] = list(dict.fromkeys(
            line.split(':', 1)[1].strip() for line in native_log.splitlines()
            if line.startswith('Warning') and ':' in line
            and line.split(':', 1)[1].strip() not in {
                '------------------------------', 'Mesh generation error summary',
                'Check the full log for details',
            } and not line.split(':', 1)[1].strip().endswith((' warning', ' warnings', ' errors'))
        ))
        if process.returncode != 0 or not (folder / 'volume.msh').is_file():
            raise ValueError(f'Gmsh meshing failed (exit {process.returncode}); see {folder / "native.log"}.')
        if not surface_state:
            raise ValueError('Gmsh generated volume without an accepted CAD surface.')
        before_points, before_faces = surface_state['points'], surface_state['faces']
        points, faces, tetrahedra = _mesh_arrays(folder / 'volume.msh', volume=True)
        if len(tetrahedra) > maximum_elements:
            raise ValueError(f'CAD mesh has {len(tetrahedra):,} elements, exceeding the requested limit of {maximum_elements:,}.')
        report.update(_same_surface(before_points, before_faces, points, faces))
        volume, checks = _tetrahedral_boundary(points, tetrahedra, faces)
        report.update(checks, material_volume_mm3=volume,
                      material_volume_relative_error=abs(volume - reference) / reference)
        if report['material_volume_relative_error'] > .01:
            raise ValueError('Gmsh tetrahedral material volume differs from CAD by more than the allowed 1%.')
        mesh = MeshTet(points.T, tetrahedra.T)
        linear_quality = attach_mesh_quality(mesh)
        if not linear_quality['solver_ready']:
            raise ValueError('CAD volume mesh contains invalid or collapsed elements.')
        report['linear_quality'] = linear_quality
        if element_order == 'C3D10':
            mesh = MeshTet2.from_mesh(mesh)
        quality = attach_mesh_quality(mesh)
        if not quality['solver_ready']:
            raise ValueError('CAD quadratic mesh contains invalid or collapsed elements.')
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise ValueError('External CAD mesher modified its STEP input.')
        report.update(passed=True, quality=quality, source_cad_step_unchanged=True,
                      element_count=len(tetrahedra), quadratic_geometry_curved=False)
        save()
        mesh.cad_meshing_report = report
        return mesh
    except Exception as error:
        report['error'] = str(error)
        save()
        try:
            detail = (folder / 'native.log').read_text(encoding='utf-8', errors='replace')[-6000:]
        except OSError:
            detail = ''
        logger.error('External CAD meshing failed: %s\nReport: %s\nNative log tail:\n%s',
                     error, json.dumps(report), detail)
        raise


__all__ = ['resolve_gmsh_executable', 'cad_tet_mesh_cli']
