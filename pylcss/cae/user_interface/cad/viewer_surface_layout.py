# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""AMRTO intermediate views with quad cells and selectable sparse regions."""
import colorsys
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


def render_surface_layout(viewer, data):
    from .viewer_simulation import build_surface_polydata
    from pylcss.cae.design_studio.topology_optimization.geometry.amrto_surface import cached_surface_layout
    layout = cached_surface_layout(data)
    if layout is None:
        return False
    grouped = data.get('visualization_mode') == 'Surface Groups'
    poly = build_surface_polydata(layout['vertices'], layout['quads'])
    if poly is None:
        return False
    group_ids = numpy_to_vtk(np.asarray(layout['group_ids'], dtype=np.int32), deep=True)
    group_ids.SetName('SurfaceGroup')
    poly.GetCellData().AddArray(group_ids)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ConsistencyOn()
    normals.SplittingOff()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    if grouped:
        count = int(layout['group_count'])
        lookup = vtk.vtkLookupTable()
        lookup.SetNumberOfTableValues(count)
        lookup.Build()
        for i in range(count):
            lookup.SetTableValue(i, *colorsys.hsv_to_rgb((i*.61803398875)%1., .42, .92), 1.)
        mapper.SetLookupTable(lookup)
        mapper.SetScalarModeToUseCellFieldData()
        mapper.SelectColorArray('SurfaceGroup')
        mapper.SetScalarRange(0, max(count-1, 1))
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    viewer.apply_cad_material(actor)
    actor.GetProperty().SetEdgeVisibility(not grouped)
    actor.GetProperty().SetEdgeColor(.06, .12, .16)
    actor.GetProperty().SetOpacity(1.)
    viewer.renderer.AddActor(actor)
    viewer.current_actor = actor
    viewer.actors.append(actor)
    if grouped:
        edges = np.asarray(layout['boundary_edges'], dtype=np.int64)
        line_data = vtk.vtkPolyData()
        line_data.SetPoints(poly.GetPoints())
        packed = np.column_stack((np.full(len(edges), 2, dtype=np.int64), edges))
        cells = vtk.vtkCellArray()
        cells.SetCells(len(edges), numpy_to_vtkIdTypeArray(packed.ravel(), deep=True))
        line_data.SetLines(cells)
        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputData(line_data)
        seams = vtk.vtkActor()
        seams.SetMapper(line_mapper)
        seams.GetProperty().SetColor(.12, .17, .22)
        seams.GetProperty().SetLineWidth(1.5)
        seams.PickableOff()
        viewer.renderer.AddActor(seams)
        viewer.actors.append(seams)
    viewer._pickable_surface_dataset = poly
    viewer._configure_mesh_face_picking_from_surface(poly)
    if viewer._section_enabled:
        viewer._update_section_cut()
    viewer._render_topopt_joint_overlays(data)
    viewer._update_scalar_bar('', 0, 1, None)
    viewer.renderer.ResetCamera()
    viewer._request_render()
    return True
