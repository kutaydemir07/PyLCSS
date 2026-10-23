# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Resolved volume meshing of closed lattice walls without a CAD conversion.

Import double-precision MSH triangles as discrete surfaces. A separately
reported analysis copy may collapse numerical short edges under a bounded
displacement; the exported lattice is unchanged. Pores are never capped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def lattice_surface_fingerprint(surface):
    digest = hashlib.sha256()
    for key, dtype in (('vertices', '<f8'), ('faces', '<i8')):
        array = np.ascontiguousarray(surface[key], dtype=dtype)
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def stress_refinement_elements(analysis):
    """Rank recovered stress variation; include all near-peak material elements.

    This is a refinement indicator, not a certified error estimator. Actual
    stress/compliance changes on the resulting mesh decide convergence.
    """
    mesh = analysis['mesh']
    stress = np.asarray(analysis['stress'], dtype=float).reshape(-1)
    dofs = mesh.dofs.element_dofs if analysis.get('element_order') == 'C3D10' else mesh.t
    values = stress[dofs]
    xyz = mesh.p[:, mesh.t].transpose(2, 1, 0)
    volumes = np.abs(np.linalg.det(np.stack((xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0],
                                           xyz[:, 3]-xyz[:, 0]), axis=-1)))/6
    indicator = np.var(values, axis=0)*volumes
    count = max(1, int(np.ceil(.25*len(indicator))))
    ids = np.argsort(-indicator, kind='stable')[:count]
    near_peak = np.flatnonzero(values.max(axis=0) >= .8*float(stress.max(initial=0.)))
    if stress.max(initial=0.) <= 0:
        near_peak = np.array([], dtype=int)
    return np.union1d(ids, near_peak).astype(int)


def reuse_lattice_mesh(base, *, element_order, max_elements, refine=False, marked_elements=None):
    """Change interpolation order or subdivide an already accepted fixed solid."""
    from copy import deepcopy
    from skfem import MeshTet, MeshTet2
    from pylcss.cae.design_studio.fem.quality import attach_mesh_quality
    report = deepcopy(getattr(base, 'surface_meshing_report', {}))
    if not report.get('passed') or element_order not in ('C3D4', 'C3D10'):
        raise ValueError('Lattice mesh reuse requires an accepted material mesh and supported element order.')
    count = int(base.t.shape[1]) * (8 if refine and marked_elements is None else 1)
    if count > max_elements:
        raise ValueError(f'Lattice refinement requires {count:,} elements; limit is {max_elements:,}.')
    corners = int(base.t.max()) + 1
    mesh = MeshTet(base.p[:, :corners].copy(), base.t.copy())
    if refine:
        mesh = mesh.refined(1 if marked_elements is None else np.asarray(marked_elements, dtype=int))
    count = int(mesh.t.shape[1])
    if count > max_elements:
        raise ValueError(f'Lattice local refinement requires {count:,} elements; limit is {max_elements:,}.')
    if element_order == 'C3D10':
        mesh = MeshTet2.from_mesh(mesh)
    quality = attach_mesh_quality(mesh)
    if not quality['solver_ready']:
        raise ValueError('Reused lattice mesh contains invalid or collapsed elements.')
    report.update(element_order=element_order, element_count=count, quality=quality,
                  accepted_volume_mesh_reused=True,
                  refinement_level=int(report.get('refinement_level', 0)) + int(refine),
                  marked_element_count=0 if marked_elements is None else len(marked_elements),
                  refinement_method=('Conforming local tetrahedral refinement from stress variation and hotspots'
                                     if marked_elements is not None else
                                     'Uniform tetrahedral subdivision of the accepted fixed faceted solid') if refine
                                     else 'Same accepted corner tetrahedra; updated interpolation order')
    mesh.surface_meshing_report = report
    mesh.boundary_selection_nodes = mesh.dofs.get_facet_dofs(mesh.boundary_facets()).all()
    return mesh


def _shells(points, faces):
    import trimesh
    mesh = trimesh.Trimesh(points, faces, process=False)
    groups = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(faces)))
    shells, tags = [], np.empty(len(faces), dtype=np.int32)
    for index, ids in enumerate(groups, 1):
        shell = trimesh.Trimesh(points, faces[ids], process=False)
        shell.remove_unreferenced_vertices()
        volume = float(shell.volume)
        if not np.isfinite(volume) or volume == 0:
            raise ValueError('A lattice shell has zero or nonfinite enclosed volume.')
        tags[ids] = index
        shells.append((index, shell, volume))
    positive = [row for row in shells if row[2] > 0]
    cavities = {tag: [] for tag, _, _ in positive}
    for tag, shell, volume in shells:
        if volume > 0:
            continue
        point = shell.vertices[:1]
        parents = [(other_volume, outer_tag) for outer_tag, outer, other_volume in positive
                   if outer.contains(point)[0]]
        if not parents:
            raise ValueError('An inward lattice cavity has no enclosing material shell.')
        cavities[min(parents)[1]].append(tag)
    return tags, cavities


def lattice_tet_mesh(surface, element_size, directory, *, element_order='C3D10',
                     max_elements=500_000, refinement_level=0, timeout_s=300.,
                     cancel_callback=None, surface_tolerance_mm=0., **_):
    import meshio
    from skfem import MeshTet, MeshTet2
    from pylcss.cae.solver_backends.execution import run_process
    from pylcss.cae.design_studio.fem.quality import attach_mesh_quality
    from ..verification._common import finite_nonnegative, validation_integer
    from .cad_gmsh_cli import (
        resolve_gmsh_executable, _closed_boundary, _screen_boundary,
        _mesh_arrays, _same_surface, _tetrahedral_boundary,
    )

    maximum = finite_nonnegative(element_size, name='Lattice element size', positive=True)
    timeout = finite_nonnegative(timeout_s, name='Lattice mesher timeout', positive=True)
    limit = validation_integer(max_elements, name='Lattice maximum elements')
    refinement = validation_integer(refinement_level, name='Lattice mesh refinement', minimum=0)
    if element_order not in ('C3D4', 'C3D10'):
        raise ValueError('Choose C3D4 or C3D10 lattice elements.')
    points, faces = np.asarray(surface['vertices'], dtype=float), np.asarray(surface['faces'])
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError('Lattice analysis requires finite indexed 3-D triangles.')
    boundary = _closed_boundary(points, faces)
    intersections = _screen_boundary(points, faces)
    from .lattice_mesh_preparation import prepare_lattice_analysis_surface
    displacement_limit = min(float(np.ptp(points, axis=0).max()) * 8 * np.finfo(np.float32).eps,
                             maximum * 1e-4)
    requested_tolerance = finite_nonnegative(surface_tolerance_mm, name='Lattice analysis surface tolerance')
    if requested_tolerance > 0:
        displacement_limit = requested_tolerance
    points, faces, preparation = prepare_lattice_analysis_surface(points, faces, displacement_limit)
    _closed_boundary(points, faces)
    prepared_intersections = _screen_boundary(points, faces)
    if preparation['material_volume_relative_change'] > .01:
        raise ValueError('Lattice analysis preparation changed material volume by more than 1%.')
    tags, volumes = _shells(points, faces)
    executable = resolve_gmsh_executable()
    if executable is None:
        raise ValueError('Lattice FEA needs the standalone Gmsh executable.')
    folder = Path(directory).resolve() / f'lattice_level_{refinement}'
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(backend='Gmsh discrete-surface volume mesh', passed=False,
                  analysis_geometry='Resolved lattice walls and struts; pores remain void',
                  source_surface_sha256=lattice_surface_fingerprint(surface),
                  source_boundary=boundary, source_intersections=intersections,
                  analysis_surface_preparation=preparation, analysis_surface_intersections=prepared_intersections,
                  requested_surface_tolerance_mm=requested_tolerance,
                  source_surface_healed=False, source_surface_smoothed=False,
                  source_CAD_used_as_analysis_geometry=False, material_shell_count=len(volumes),
                  cavity_shell_count=len(np.unique(tags)) - len(volumes),
                  refinement_level=refinement, refinement_method='Uniform tetrahedral subdivision of the fixed faceted solid',
                  requested_maximum_size_mm=maximum, element_order=element_order)
    try:
        meshio.write(folder/'surface_input.msh', meshio.Mesh(points, [('triangle', faces)],
            cell_data={'gmsh:geometrical': [tags], 'gmsh:physical': [tags]}),
            file_format='gmsh22', binary=False)
        source_hash = hashlib.sha256((folder/'surface_input.msh').read_bytes()).hexdigest()
        lines = ['General.AbortOnError=4;', 'General.NoPopup=1;', 'General.NumThreads=4;',
                 'Mesh.Algorithm3D=1;', 'Mesh.OptimizeNetgen=1;', 'Mesh.AngleToleranceFacetOverlap=.000001;',
                 'Mesh.MeshOnlyEmpty=1;', 'Mesh.SaveAll=1;', 'Mesh.ElementOrder=1;',
                 'Mesh.MshFileVersion=2.2;', 'Mesh.Binary=0;', f'Mesh.MeshSizeMax={maximum:.17g};',
                 'Merge "surface_input.msh";']
        lines.extend(f'Surface Loop({1000+int(tag)})={{{int(tag)}}};' for tag in np.unique(tags))
        for tag, holes in volumes.items():
            loops = ','.join(str(1000+t) for t in (tag, *holes))
            lines.append(f'Volume({1000+tag})={{{loops}}};')
        lines.extend(('Mesh 3;', 'Save "volume.msh";'))
        (folder/'mesh.geo').write_text('\n'.join(lines)+'\n', encoding='ascii')
        (folder/'volume.msh').unlink(missing_ok=True)
        result = run_process([executable, 'mesh.geo', '-parse_and_exit', '-nopopup', '-v', '4'],
            cwd=folder, timeout_s=timeout, stdout_file=folder/'native.log', cancel_callback=cancel_callback)
        if result.returncode != 0 or not (folder/'volume.msh').is_file():
            raise ValueError(f'Gmsh could not fill the lattice material (exit {result.returncode}).')
        nodes, triangles, tets = _mesh_arrays(folder/'volume.msh', volume=True)
        report.update(_same_surface(points, faces, nodes, triangles))
        volume, checks = _tetrahedral_boundary(nodes, tets, triangles)
        checks['actual_tetrahedral_exterior_equals_analysis_surface'] = checks.pop(
            'actual_tetrahedral_exterior_equals_native_CAD_surface')
        report.update(checks, material_volume_mm3=volume,
                      material_volume_relative_error=abs(volume-boundary['oriented_volume_mm3'])/boundary['oriented_volume_mm3'])
        if report['material_volume_relative_error'] > .01:
            raise ValueError('Lattice tetrahedral volume differs from its closed surface by more than 1%.')
        if len(tets) * 8**refinement > limit:
            raise ValueError(f'Lattice mesh/refinement requires {len(tets)*8**refinement:,} elements; limit is {limit:,}.')
        mesh = MeshTet(nodes.T, tets.T)
        if refinement:
            mesh = mesh.refined(refinement)
        if element_order == 'C3D10':
            mesh = MeshTet2.from_mesh(mesh)
        quality = attach_mesh_quality(mesh)
        if not quality['solver_ready']:
            raise ValueError('Lattice volume mesh contains invalid or collapsed elements.')
        if hashlib.sha256((folder/'surface_input.msh').read_bytes()).hexdigest() != source_hash:
            raise ValueError('The external mesher modified the lattice surface input.')
        report.update(passed=True, element_count=int(mesh.t.shape[1]), quality=quality,
                      native_warnings=[line for line in (folder/'native.log').read_text(errors='replace').splitlines()
                                       if line.startswith('Warning')])
        mesh.surface_meshing_report = report
        mesh.boundary_selection_nodes = mesh.dofs.get_facet_dofs(mesh.boundary_facets()).all()
        return mesh
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (folder/'meshing_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
