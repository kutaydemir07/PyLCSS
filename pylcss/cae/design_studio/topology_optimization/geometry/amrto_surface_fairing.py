# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded spline fairing with fixed shared edges and independent admission.

The quadratic parametric thin-plate proxy is supplemented by one-sided first
derivative jumps at C0 knots. Integrating only open knot spans misses those
jumps entirely. This engineering extension is not a global G1 certificate.
"""
from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve


def fairing_displacement_budget(layout,payload):
    """Use source uncertainty for interiors, independently of edge matching."""
    acceptance=layout.fitting_report.get('source_fit_acceptance',{})
    limit=float(getattr(layout,'solid_source_fit_limit',acceptance.get('maximum_deviation_mm',0.)))
    measured=float(acceptance.get('measured_deviation_mm',limit))
    margin=max(0.,limit-measured)
    budget=min(float(getattr(layout,'solid_fit_tolerance',0.)),.5*margin)
    basis='Boundary budget fallback; no source voxel spacing available'
    field=payload.get('recovery_density')
    if field is None:field=payload.get('density')
    bounds=payload.get('bounds')
    spacing=None
    if field is not None and np.ndim(field)==3 and bounds is not None:
        low,high=(bounds['min'],bounds['max']) if isinstance(bounds,dict) else bounds
        spacing=(np.asarray(high,dtype=float)-np.asarray(low,dtype=float))/np.asarray(np.shape(field))
        if np.isfinite(spacing).all() and np.all(spacing>0):
            budget=min(.5*float(spacing.min()),.5*margin)
            basis='Half the finest source voxel edge, capped by half the remaining source-deviation allowance'
    return budget,{'maximum_surface_displacement_mm':float(budget),'basis':basis,
        'source_voxel_spacing_mm':None if spacing is None else spacing.tolist(),
        'source_fit_limit_mm':limit,'initial_source_fit_deviation_mm':measured,
        'shared_boundary_geometry_unchanged':True,'source_admission_and_volume_checks_required':True}


def _axis_grams(knots, degree, count):
    unique = np.unique(knots)
    sites, weights = [], []
    gauss, quadrature = np.polynomial.legendre.leggauss(degree+1)
    for lo, hi in zip(unique[:-1], unique[1:]):
        sites.extend(.5*((hi-lo)*gauss+hi+lo))
        weights.extend(.5*(hi-lo)*quadrature)
    basis = BSpline(knots, np.eye(count), degree)
    grams = []
    for derivative in range(3):
        values = sparse.csr_matrix(basis(sites, nu=derivative))
        grams.append(values.T@sparse.diags(weights)@values)
    jump = sparse.csr_matrix((count, count))
    for index, knot in enumerate(unique[1:-1], 1):
        if np.count_nonzero(np.asarray(knots) == knot) < degree:
            continue
        epsilon = min(knot-unique[index-1], unique[index+1]-knot)*1e-7
        difference = basis(knot+epsilon, nu=1)-basis(knot-epsilon, nu=1)
        row = sparse.csr_matrix(difference[None])
        jump += (row.T@row)/min(knot-unique[index-1], unique[index+1]-knot)
    grams[2] += jump
    return grams


def fairness_matrix(surface):
    """Physical-length-scaled parametric bending energy, including C0 jumps."""
    controls = np.asarray(surface.ctrlpts).reshape(surface.ctrlpts_size_u, surface.ctrlpts_size_v, 3)
    gu = _axis_grams(surface.knotvector_u, surface.degree_u, controls.shape[0])
    gv = _axis_grams(surface.knotvector_v, surface.degree_v, controls.shape[1])
    u = float(np.mean(np.sum(np.linalg.norm(np.diff(controls, axis=0), axis=2), axis=0)))
    v = float(np.mean(np.sum(np.linalg.norm(np.diff(controls, axis=1), axis=2), axis=1)))
    if min(u, v) <= 0:
        raise ValueError('Fairing needs nondegenerate spline directions.')
    reference = min(u, v)
    matrix = (sparse.kron(gu[2], gv[0])*(reference/u)**4
              +2*sparse.kron(gu[1], gv[1])*(reference**2/(u*v))**2
              +sparse.kron(gu[0], gv[2])*(reference/v)**4).tocsr()
    diagonal = matrix.diagonal()
    matrix /= max(float(np.median(diagonal[diagonal > 0])), 1e-30)
    return matrix


def internal_normal_matrix(surface):
    """Linear tangent-plane targets on both sides of each internal C0 line."""
    from .amrto_seam_fairing import _basis, _normal
    controls = np.asarray(surface.ctrlpts).reshape(surface.ctrlpts_size_u,surface.ctrlpts_size_v,3)
    free_mask=np.zeros(controls.shape[:2],bool);free_mask[1:-1,1:-1]=True
    free=np.flatnonzero(free_mask.ravel())
    axes=[np.unique(surface.knotvector_u),np.unique(surface.knotvector_v)]
    rows=[]
    for axis,(knots,degree) in enumerate(((surface.knotvector_u,surface.degree_u),(surface.knotvector_v,surface.degree_v))):
        unique=axes[axis];other=axes[1-axis]
        sites=np.unique(np.concatenate([lo+np.array([.2113248654,.5,.7886751346])*(hi-lo)
                                       for lo,hi in zip(other[:-1],other[1:])]))
        for index,knot in enumerate(unique[1:-1],1):
            if np.count_nonzero(np.asarray(knots)==knot)<degree:continue
            epsilon=min(knot-unique[index-1],unique[index+1]-knot)*1e-7
            evaluated=[]
            for value in (knot-epsilon,knot+epsilon):
                uv=np.column_stack((np.full(len(sites),value),sites))
                if axis:uv=uv[:,::-1]
                _,du,dv,poles=_basis(surface,uv)
                evaluated.append((du,dv,_normal(du,dv,poles)))
            left,right=evaluated
            sign=np.where(np.einsum('ij,ij->i',left[2],right[2])>=0,1.,-1.)
            target=left[2]+sign[:,None]*right[2]
            tangent=(left[1-axis]@controls.reshape(-1,3))
            tangent/=np.linalg.norm(tangent,axis=1)[:,None]
            target-=np.einsum('ij,ij->i',target,tangent)[:,None]*tangent
            target/=np.linalg.norm(target,axis=1)[:,None]
            for current in evaluated:
                derivative=current[axis]
                sensitivity=np.linalg.norm(derivative[:,free],axis=1)
                keep=sensitivity>1e-12
                block=(derivative[keep,:,None]*target[keep,None,:])/sensitivity[keep,None,None]
                rows.append(sparse.csr_matrix(block.reshape(sum(keep),-1)))
    return sparse.vstack(rows,format='csr') if rows else sparse.csr_matrix((0,3*controls.shape[0]*controls.shape[1]))


def _pole_normals(surface,controls):
    """Reference tangent planes at Greville sites, without a dense 2D basis."""
    matrices=[]
    for knots,degree,count in ((surface.knotvector_u,surface.degree_u,controls.shape[0]),
                                (surface.knotvector_v,surface.degree_v,controls.shape[1])):
        sites=np.array([np.mean(knots[i+1:i+degree+1]) for i in range(count)])
        basis=BSpline(knots,np.eye(count),degree)
        matrices.append((basis(sites),basis(sites,nu=1)))
    du=np.einsum('ai,ijc,bj->abc',matrices[0][1],controls,matrices[1][0],optimize=True)
    dv=np.einsum('ai,ijc,bj->abc',matrices[0][0],controls,matrices[1][1],optimize=True)
    normals=np.cross(du,dv);length=np.linalg.norm(normals,axis=2)
    return np.divide(normals,length[:,:,None],out=np.zeros_like(normals),where=length[:,:,None]>1e-14)


def fair_surface_interior(layout, maximum_displacement, *, strength=1., normal_targets=False):
    """One fixed-basis proposal; keep source, stock, orientation and edges checked."""
    from .amrto_continuity import inspect_internal_spline_continuity
    from .amrto_fit_orientation import assess_fit_orientation
    from .amrto_nurbs_fitting import _surface_map_residual
    from .amrto_shared_stock import control_hull_in_box
    original = layout.nurbs_surface
    report = dict(method='Bounded parametric thin-plate and C0-jump fairing', applied=False,
        shared_boundary_poles_unchanged=True, global_G1_continuity_certified=False)
    weights = getattr(original, 'weights', None)
    if weights is not None and not np.all(np.asarray(weights) == 1.):
        raise ValueError('Bounded fairing requires a unit-weight spline.')
    if maximum_displacement <= 0 or not np.isfinite(maximum_displacement) or strength <= 0 or not np.isfinite(strength):
        raise ValueError('Fairing requires finite positive displacement and strength.')
    controls = np.asarray(original.ctrlpts).reshape(original.ctrlpts_size_u, original.ctrlpts_size_v, 3)
    if controls.shape[0]*controls.shape[1] > 4096:
        return original,{**report,'reason':'Fairing control-count budget exceeded'}
    interior = np.zeros(controls.shape[:2], dtype=bool);interior[1:-1, 1:-1] = True
    free = np.flatnonzero(interior.ravel());flat = controls.reshape(-1, 3)
    if not len(free):
        return original, {**report, 'reason':'No free interior poles'}
    matrix = fairness_matrix(original)
    if normal_targets:
        normals=internal_normal_matrix(original)
        if not normals.shape[0] or not normals.nnz:
            return original,{**report,'reason':'No internal C0 normal targets'}
        components=(3*free[:,None]+np.arange(3)).ravel()
        local=normals[:,components]
        delta=spsolve(sparse.eye(len(components),format='csc')+strength*(local.T@local).tocsc(),
                      -strength*(local.T@(normals@flat.ravel()))).reshape(-1,3)
    else:
        delta = spsolve(sparse.eye(len(free), format='csc')+strength*matrix[free][:,free].tocsc(),
                        -strength*(matrix[free]@flat))
    if not np.isfinite(delta).all():
        return original, {**report, 'reason':'Nonfinite fairing solve'}
    # Tangential pole movement can merely distort a planar parameterization;
    # it does not remove physical waviness and destabilizes exact CAD trims.
    # Restrict the proposal to the reference surface-normal directions.
    normals=_pole_normals(original,controls).reshape(-1,3)[free]
    delta=np.einsum('ij,ij->i',delta,normals)[:,None]*normals
    bounds = getattr(layout, 'source_box_bounds', None)
    if bounds is not None:
        low, high = np.asarray(bounds)
        factors=np.ones(len(free))
        for axis in range(3):
            positive=delta[:,axis]>0;negative=delta[:,axis]<0
            factors[positive]=np.minimum(factors[positive],(high[axis]-flat[free[positive],axis])/delta[positive,axis])
            factors[negative]=np.minimum(factors[negative],(low[axis]-flat[free[negative],axis])/delta[negative,axis])
        delta*=np.clip(factors,0.,1.)[:,None]
    if np.linalg.norm(delta,axis=1).max(initial=0.) <= max(float(np.ptp(flat,axis=0).max())*1e-12,1e-12):
        return original,{**report,'reason':'No physical normal displacement required'}
    scale = min(1., maximum_displacement/max(float(np.linalg.norm(delta,axis=1).max()),1e-30))
    if scale <= 0:
        return original, {**report,'reason':'No movement available inside source stock'}
    poles = flat.copy();poles[free] += scale*delta
    candidate = deepcopy(original);candidate.set_ctrlpts(poles.tolist(),*controls.shape[:2])
    before = inspect_internal_spline_continuity(original)
    after = inspect_internal_spline_continuity(candidate)
    orientation = assess_fit_orientation(layout,candidate)
    deviation = _surface_map_residual(layout,candidate,grid_size=73)
    limit = getattr(layout,'solid_source_fit_limit',maximum_displacement)
    stock = control_hull_in_box(poles,bounds) if bounds is not None else {'accepted':True}
    energy_before=float(np.sum(flat*(matrix@flat)));energy_after=float(np.sum(poles*(matrix@poles)))
    normal_before=sum((line['maximum_sampled_tangent_plane_jump_degrees'] or 0.)**2 for line in before['lines'])
    normal_after=sum((line['maximum_sampled_tangent_plane_jump_degrees'] or 0.)**2 for line in after['lines'])
    accepted=bool(orientation['accepted'] and deviation <= limit and stock['accepted']
        and (normal_after < normal_before-1e-10 if normal_targets else
             energy_after < energy_before-1e-12*max(energy_before,1.))
        and after['maximum_sampled_tangent_plane_jump_degrees'] <= before['maximum_sampled_tangent_plane_jump_degrees']+1e-5)
    report.update(applied=accepted,maximum_surface_displacement_bound_mm=float(np.linalg.norm(poles-flat,axis=1).max()),
        tangential_reparameterization_excluded=True,
        internal_normal_targets=normal_targets,
        bending_proxy_before=energy_before,bending_proxy_after=energy_after,
        internal_continuity_before=before,internal_continuity_after=after,
        source_deviation_mm=deviation,source_limit_mm=float(limit),orientation=orientation,stock=stock)
    return (candidate if accepted else original),report
