# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Spatial peak locations for reviewing the actual, unfiltered FEA stresses."""
import numpy as np


def stress_hotspots(analysis, separation_mm, maximum=5):
    """Retain spatially separated nodal maxima, with local mesh dimensions."""
    mesh=analysis['mesh']
    stress=np.asarray(analysis['stress'],dtype=float).reshape(-1)
    points=np.asarray(mesh.p,dtype=float).T
    if stress.shape!=(len(points),) or not np.isfinite(stress).all():
        raise ValueError('Hotspot coordinates require finite nodal stresses on the analysis mesh.')
    conn=mesh.dofs.element_dofs if analysis.get('element_order')=='C3D10' else mesh.t
    selected=[];rows=[]
    for node in np.argsort(-stress,kind='stable'):
        point=points[node]
        if any(np.linalg.norm(point-p)<separation_mm for p in selected):continue
        selected.append(point)
        incident=np.flatnonzero(np.any(conn==node,axis=0))
        corners=points[mesh.t[:,incident].T]
        if len(corners):
            edges=np.linalg.norm(corners[:,[0,0,0,1,1,2]]-corners[:,[1,2,3,2,3,3]],axis=2)
            edge_bounds=[float(edges.min()),float(edges.max())]
        else:edge_bounds=None
        rows.append({'node_index':int(node),'coordinates_mm':point.tolist(),
                     'von_mises_mpa':float(stress[node]),'incident_element_count':len(incident),
                     'incident_corner_edge_range_mm':edge_bounds})
        if len(rows)>=maximum:break
    return {'method':'Spatially separated maxima of the unfiltered reported nodal stress',
            'separation_mm':float(separation_mm),'stress_location':analysis.get('stress_location'),
            'stress_reduced_or_averaged':False,'peaks':rows}
