# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM static linear solver node."""
import numpy as np
import logging
import os
import tempfile
import sys
import contextlib
from scipy.spatial import cKDTree
import skfem
from skfem import *
from skfem.helpers import sym_grad, ddot, trace
try:
    from simpleeval import simple_eval
except ImportError:
    simple_eval = None
from pylcss.config import simulation_config
from pylcss.cad.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)
from pylcss.cad.nodes.fem._helpers import (
    lam_lame, tr,
    _assemble_traction_force, _assemble_pressure_force,
    _find_matching_boundary_facets,
)

class SolverNode(CadQueryNode):
    """Solves the FEA problem."""
    __identifier__ = 'com.cad.sim.solver'
    NODE_NAME = 'FEA Solver'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        self.add_input('material', color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100), multi_input=True)
        self.add_input('loads', color=(255, 255, 0), multi_input=True)
        self.add_output('results', color=(0, 255, 255))
        self.create_property('visualization', 'Von Mises Stress', widget_type='combo',
                             items=['Von Mises Stress', 'Displacement'])
        # Deformation scale for visualization (Auto scales so peak disp ≈ 5% of bbox)
        self.create_property('deformation_scale', 'Auto', widget_type='combo',
                             items=['Auto', '1x', '5x', '10x', '50x', '100x', '200x'])

    def run(self):
        print("FEA Solver: Node 'run' called.")
        mesh = self.get_input_value('mesh', None)
        material = self.get_input_value('material', None)
        
        # Helper to flatten inputs
        def flatten_inputs(inputs):
            res = []
            for item in inputs:
                if isinstance(item, list):
                    res.extend(flatten_inputs(item))
                elif item is not None:
                    res.append(item)
            return res

        constraint_list = flatten_inputs(self.get_input_list('constraints'))
        load_list = flatten_inputs(self.get_input_list('loads'))
        
        print(f"FEA Solver: Inputs - Mesh: {mesh is not None}, Mat: {material is not None}, Constraints: {len(constraint_list)}, Loads: {len(load_list)}")
        
        if not (mesh and material and constraint_list and load_list):
            print("FEA Solver: Missing required inputs. Aborting.")
            return None

        # 1. Define Element and Basis (Vector)
        # CHANGE: Use P2 (Quadratic) elements for accuracy (prevents locking)
        print("FEA Solver: Initializing P2 Basis (Quadratic)...")
        e = ElementVector(ElementTetP2())
        basis = Basis(mesh, e)
        print(f"FEA Solver: Basis Initialized. Total DOFs: {basis.N}")

        # 2. Define Physics (Linear Elasticity)
        E_mat   = material['E']
        nu_mat  = float(material.get('nu', material.get('poissons_ratio', 0.3))) # Use 'nu' or 'poissons_ratio'
        rho_mat = float(material.get('rho', 7.85e-9))   # t/mm³ — for gravity body force

        # ── Incompressibility guard ───────────────────────────────────────────
        # Lamé λ = Eν / ((1+ν)(1-2ν)).  At ν = 0.5 the denominator is zero;
        # even at ν = 0.499 λ ≫ μ by 2–3 orders of magnitude, making K
        # almost singular and producing near-zero displacements (volumetric
        # locking).  This displacement-only formulation cannot represent truly
        # incompressible materials — that requires a mixed u-p (displacement-
        # pressure) element.  Cap ν conservatively and warn the user.
        _NU_MAX_DISP = 0.495
        if nu_mat > _NU_MAX_DISP:
            _nu_warn = (
                f"FEA Solver: WARNING — Poisson's ratio ν={nu_mat:.4f} is at or "
                f"above the incompressibility limit.\n"
                f"  Capping to ν={_NU_MAX_DISP} for numerical stability.\n"
                "  At ν≥0.5 the Lamé parameter λ = Eν/((1+ν)(1-2ν)) diverges,\n"
                "  making K singular (volumetric locking).\n"
                "  Accurate modelling of near-incompressible materials (rubber,\n"
                "  biological tissue, etc.) requires a mixed u-p formulation\n"
                "  which this displacement-only solver does not implement."
            )
            print(_nu_warn)
            logger.warning(_nu_warn)
            nu_mat = _NU_MAX_DISP

        # Lame parameters
        lam_val, mu_val = lam_lame(E_mat, nu_mat)
        
        # Create constant fields for material properties
        # We need a scalar basis for parameters
        basis0 = basis.with_element(ElementTetP0())
        lam_field = basis0.zeros() + lam_val
        mu_field = basis0.zeros() + mu_val
        
        lam_interp = basis0.interpolate(lam_field)
        mu_interp = basis0.interpolate(mu_field)

        @BilinearForm
        def stiffness(u, v, w):
            def epsilon(w):
                return sym_grad(w)
            E = epsilon(u)
            D = epsilon(v)
            return 2.0 * w['mu'] * ddot(E, D) + w['lam'] * tr(E) * tr(D)

        # 3. Assemble Stiffness Matrix
        print(f"FEA Solver: Assembling Stiffness Matrix (DOF: {basis.N})...")
        K = stiffness.assemble(basis, lam=lam_interp, mu=mu_interp)
        print("FEA Solver: Assembly Complete.")

        # 4. Apply Boundary Conditions
        x, y, z = mesh.p

        fixed_dofs = np.array([], dtype=int)
        # Prescribed displacement vector for non-zero Displacement BCs.
        # condense(K, f, x=u_prescribed, D=fixed_dofs) enforces u[fixed_dofs] = u_prescribed[fixed_dofs].
        u_prescribed = np.zeros(basis.N)

        for constraint in constraint_list:
            if not constraint: continue

            fixed_dof_indices = constraint.get('fixed_dofs', [0, 1, 2])
            disp_vals = constraint.get('displacement', None)  # [dx, dy, dz] or None
            c_type = constraint.get('type', 'Fixed')

            try:
                # Handle geometry-based selection (either single 'geometry' or list 'geometries')
                geoms = constraint.get('geometries', [constraint.get('geometry')])
                geoms = [g for g in geoms if g is not None]

                if geoms:
                    tolerance = 1.5

                    # Use skfem's native DOF locator so that mid-edge / face-centre
                    # DOFs introduced by quadratic (P2) elements are also captured.
                    # FIX #3: Use default-argument capture (_geoms=geoms) to avoid
                    # the Python late-binding closure bug — without this, all
                    # iterations share the same 'geoms' reference (last loop value).
                    from cadquery import Vector as _CQVector
                    def _is_on_face(pts, _geoms=geoms, _tol=tolerance):
                        # pts has shape (3, N); return a bool mask of length N.
                        px, py, pz = pts[0], pts[1], pts[2]
                        mask = np.zeros(len(px), dtype=bool)
                        for i in range(len(px)):
                            pt = _CQVector(float(px[i]), float(py[i]), float(pz[i]))
                            for g in _geoms:
                                try:
                                    if g.distanceTo(pt) <= _tol:
                                        mask[i] = True
                                        break
                                except Exception:
                                    try:
                                        bb = g.BoundingBox()
                                        if (bb.xmin - _tol <= px[i] <= bb.xmax + _tol and
                                                bb.ymin - _tol <= py[i] <= bb.ymax + _tol and
                                                bb.zmin - _tol <= pz[i] <= bb.zmax + _tol):
                                            mask[i] = True
                                            break
                                    except Exception:
                                        pass
                        return mask

                    facet_dofs = basis.get_dofs(_is_on_face)
                    n_found = sum(len(facet_dofs.nodal[f'u^{k+1}']) for k in range(3))
                    # Add facet DOFs for P2 elements
                    if hasattr(facet_dofs, 'facet'):
                        n_found += sum(len(facet_dofs.facet.get(f'u^{k+1}', [])) for k in range(3))
                    print(f"SolverNode: found {n_found} DOFs matching face geometry for {c_type}")

                    for dof_idx in fixed_dof_indices:
                        # Collect both nodal DOFs (vertices) and facet DOFs (mid-edge/face for P2)
                        dofs_nodal = facet_dofs.nodal[f'u^{dof_idx+1}']
                        dofs_facet = facet_dofs.facet.get(f'u^{dof_idx+1}', np.array([], dtype=int))
                        dofs = np.concatenate([dofs_nodal, dofs_facet])
                        fixed_dofs = np.union1d(fixed_dofs, dofs)
                        if disp_vals is not None:
                            u_prescribed[dofs] = float(disp_vals[dof_idx])
                    
                elif 'condition' in constraint and constraint['condition']:
                    # LEGACY: String-based constraint (fallback)
                    cond_str = constraint['condition']
                    
                    # Secure evaluation using simpleeval
                    if simple_eval is not None:
                        # Define allowed names and functions for safe evaluation
                        names = {'x': x, 'y': y, 'z': z}
                        functions = {'sin': np.sin, 'cos': np.cos, 'abs': np.abs, 'sqrt': np.sqrt}
                        
                        def constraint_func(p):
                            x_val, y_val, z_val = p
                            names.update({'x': x_val, 'y': y_val, 'z': z_val})
                            return simple_eval(cond_str, names=names, functions=functions)
                    else:
                        # Fallback to restricted eval if simpleeval not available
                        def constraint_func(p):
                            x_val, y_val, z_val = p
                            return eval(cond_str, {'x': x_val, 'y': y_val, 'z': z_val, 'np': np})
                    
                    facet_dofs = basis.get_dofs(constraint_func)
                    matched = False
                    for dof_idx in fixed_dof_indices:
                        dofs = facet_dofs.nodal[f'u^{dof_idx+1}']
                        if len(dofs) > 0:
                            matched = True
                            fixed_dofs = np.union1d(fixed_dofs, dofs)

                    if not matched:
                        try:
                            if simple_eval is not None:
                                condition_results = []
                                for i in range(len(x)):
                                    names = {
                                        'x': float(x[i]),
                                        'y': float(y[i]),
                                        'z': float(z[i]),
                                    }
                                    functions = {
                                        'sin': np.sin,
                                        'cos': np.cos,
                                        'abs': abs,
                                        'sqrt': np.sqrt,
                                    }
                                    condition_results.append(
                                        bool(simple_eval(cond_str, names=names, functions=functions))
                                    )
                                matching_nodes = np.where(condition_results)[0]
                            else:
                                matching_nodes = np.where(
                                    eval(cond_str, {'x': x, 'y': y, 'z': z, 'np': np})
                                )[0]

                            for dof_idx in fixed_dof_indices:
                                fixed_dofs = np.union1d(
                                    fixed_dofs,
                                    basis.nodal_dofs[dof_idx, matching_nodes],
                                )
                        except Exception as node_exc:
                            logger.warning(
                                f"FEA Solver: Node-based constraint fallback failed: {node_exc}"
                            )
                    
            except Exception as e:
                logger.warning(f"FEA Solver: Constraint processing error: {e}")

        fixed_dofs = fixed_dofs.astype(int)

        # ── Rigid-Body-Motion (RBM) Singularity Guard ────────────────────────────
        # In 3-D, a body has 6 rigid-body modes: 3 translations + 3 rotations.
        # The assembled stiffness matrix K stays singular until ALL six modes are
        # suppressed by boundary conditions.  The minimum requirement is:
        #   • At least one DOF constrained in each global direction (X, Y, Z)
        #     → prevents pure translational rigid-body motion.
        #   • Total of ≥ 6 fixed DOFs distributed across non-collinear,
        #     non-coplanar nodes → prevents pure rotational modes as well.
        # This pre-check catches the most common user mistakes (forgetting a
        # constraint, pinning only a single node, etc.) and emits an actionable
        # warning BEFORE the solver call so the user sees a clear message
        # rather than a cryptic LinAlgError or NaN displacement field.
        _rbm_issues: list[str] = []
        if len(fixed_dofs) == 0:
            _rbm_issues.append(
                "No DOFs are constrained — the model is completely free to move "
                "and K will be singular."
            )
        else:
            if len(fixed_dofs) < 6:
                _rbm_issues.append(
                    f"Only {len(fixed_dofs)} DOF(s) are constrained. "
                    "At least 6 are required to suppress all 3 translational "
                    "and 3 rotational rigid-body modes in 3-D."
                )
            # For skfem vector-DOF layout the Cartesian direction index
            # equals  DOF_global mod 3  (0 → X, 1 → Y, 2 → Z).
            _constrained_dirs = set(int(d) % 3 for d in fixed_dofs)
            _missing_dirs = {0, 1, 2} - _constrained_dirs
            if _missing_dirs:
                _dir_labels = {0: 'X', 1: 'Y', 2: 'Z'}
                _missing_str = ', '.join(_dir_labels[d] for d in sorted(_missing_dirs))
                _rbm_issues.append(
                    f"No constraint applied in direction(s): {_missing_str}. "
                    "The structure can translate freely along these axes — K will be singular."
                )
        if _rbm_issues:
            _rbm_msg = (
                "FEA Solver: WARNING — Possible rigid-body motion (RBM) detected.\n"
                "  The stiffness matrix K may be singular and the linear solve will fail.\n"
                "  Most likely cause: insufficient boundary conditions.\n"
                "  Issues found:\n"
                + "".join(f"    • {w}\n" for w in _rbm_issues)
                + "  Fix: add constraints on distinct, non-collinear/non-coplanar nodes\n"
                  "  that collectively suppress all 6 rigid-body DOFs."
            )
            print(_rbm_msg)
            logger.warning(_rbm_msg)

        # 5. Apply Loads
        f = np.zeros(basis.N)
        
        try:
            for load in load_list:
                if not load: continue
                
                if load['type'] == 'pressure':
                    # Pressure is traction = p n(x) over the loaded surface.
                    # Use mesh facet normals so curved faces get the correct
                    # distributed resultant and moment.
                    pressure     = load['pressure']
                    all_faces    = load.get('geometries', None)
                    if not all_faces:
                        single = load.get('geometry', None)
                        all_faces = [single] if single is not None else []

                    f_pressure = _assemble_pressure_force(mesh, basis, all_faces, pressure)
                    if f_pressure is not None:
                        f += f_pressure
                        print(f"FEA Solver: Pressure {pressure} MPa applied via facet-normal pressure integration.")
                    else:
                        logger.error(
                            "FEA Solver: Pressure FacetBasis assembly failed — "
                            "no facets matched loaded geometry.  "
                            "Check that the geometry face lies on the mesh boundary."
                        )
                                
                elif load['type'] == 'force':
                    # Support for geometry-based force selection
                    geoms = load.get('geometries', [load.get('geometry')])
                    geoms = [g for g in geoms if g is not None]
                    load_vec = load['vector']

                    if geoms:
                        # ----------------------------------------------------------------
                        # PRIMARY PATH: proper Neumann BC via FacetBasis integration.
                        # This distributes the force area-weighted over the loaded surface
                        # facets, which is mathematically correct for unstructured meshes.
                        # ----------------------------------------------------------------
                        f_traction = _assemble_traction_force(mesh, basis, geoms, load_vec)
                        if f_traction is not None:
                            f += f_traction
                            print(f"FEA Solver: Force {load_vec} applied via FacetBasis traction integration.")
                        else:
                            # FIX #4: Surface the load failure visibly — do NOT silently skip.
                            # Equal nodal distribution would give 'bed of nails' stress spikes
                            # on unstructured meshes and is worse than failing loudly.
                            _trac_err = (
                                f"FEA Solver: FacetBasis traction assembly FAILED for force {load_vec}.\n"
                                "  No load was applied — the simulation would run with zero external force\n"
                                "  (producing near-zero results that look deceptively 'converged').\n"
                                "  Likely causes:\n"
                                "    • Selected face does not coincide with the mesh boundary.\n"
                                "    • Mesh element size larger than the face — no boundary facets found.\n"
                                "    • Tolerance (1.5 mm) too small for the geometry scale.\n"
                                "  Fix: verify SelectFaceNode → LoadNode geometry and mesh resolution."
                            )
                            logger.error(_trac_err)
                            print(_trac_err)
                            self.set_error(_trac_err)
                            return None

                    elif 'condition' in load and load['condition']:
                        # LEGACY: Handle force loads via condition string
                        load_cond = load['condition']
                        matching_nodes_indices = []
                        # Secure evaluation using simpleeval
                        if simple_eval is not None:
                            try:
                                x_arr, y_arr, z_arr = np.asarray(x), np.asarray(y), np.asarray(z)
                                condition_results = []
                                for i in range(len(x_arr)):
                                    names = {'x': float(x_arr[i]), 'y': float(y_arr[i]), 'z': float(z_arr[i])}
                                    functions = {'sin': np.sin, 'cos': np.cos, 'abs': abs, 'sqrt': np.sqrt}
                                    result = simple_eval(load_cond, names=names, functions=functions)
                                    condition_results.append(bool(result))
                                matching_nodes_indices = np.where(condition_results)[0]
                            except Exception:
                                matching_nodes_indices = np.where(eval(load_cond, {'x': x, 'y': y, 'z': z, 'np': np}))[0]
                        else:
                            matching_nodes_indices = np.where(eval(load_cond, {'x': x, 'y': y, 'z': z, 'np': np}))[0]
                        n_load_nodes = len(matching_nodes_indices)
                        if n_load_nodes > 0:
                            fx_total, fy_total, fz_total = load_vec
                            nodal_dofs = basis.nodal_dofs
                            weight = 1.0 / n_load_nodes
                            for node_idx in matching_nodes_indices:
                                f[nodal_dofs[0, node_idx]] += fx_total * weight
                                f[nodal_dofs[1, node_idx]] += fy_total * weight
                                f[nodal_dofs[2, node_idx]] += fz_total * weight

                elif load['type'] == 'gravity':
                    # ── Gravity body force:  ∫ ρ g⃗ · v dΩ ─────────────────────────
                    # Correctly integrates self-weight over the whole volume.
                    # Unit system: rho [t/mm³] × g [mm/s²] = force/volume [N/mm³]
                    _g_accel   = float(load.get('accel', 9810.0))     # mm/s²
                    _g_dir_str = load.get('direction', '-Y')
                    _dir_map   = {
                        '-Y': [0., -1.,  0.], '+Y': [0.,  1.,  0.],
                        '-Z': [0.,  0., -1.], '+Z': [0.,  0.,  1.],
                        '-X': [-1., 0.,  0.], '+X': [1.,  0.,  0.],
                    }
                    _gvec = _dir_map.get(_g_dir_str, [0., -1., 0.])
                    _fvol = rho_mat * _g_accel                        # N/mm³

                    @LinearForm
                    def _gravity_form(v, w):
                        return _fvol * (_gvec[0]*v[0] + _gvec[1]*v[1] + _gvec[2]*v[2])

                    f += _gravity_form.assemble(basis)
                    print(
                        f"FEA Solver: Gravity body force assembled "
                        f"(ρ={rho_mat:.3e} t/mm³  g={_g_accel:.1f} mm/s²  "
                        f"dir={_g_dir_str}  f/V={_fvol:.3e} N/mm³)"
                    )

        except Exception:
            pass

        # ── Pre-solve scalars (safe defaults survive a failed solve) ──────────────
        _max_disp_val   = 0.0
        _disp_scale_val = 1.0

        # 6. Solve
        try:
            print(f"FEA Solver: Starting Linear Solve (Fixed DOFs: {len(fixed_dofs)})...")
            # Pass u_prescribed so that Displacement BCs with non-zero values are enforced
            # correctly.  For Fixed/Roller/Pinned BCs u_prescribed[dofs] == 0.0, so this
            # is backward-compatible with the zero-displacement case.
            u = solve(*condense(K, f, x=u_prescribed, D=fixed_dofs))
            _max_disp_val = np.max(np.abs(u))
            print(f"FEA Solve Complete. Max Displacement: {_max_disp_val:.6e}")

            # ── Deformation display scale ────────────────────────────────────
            # 'Auto' scales so max displacement renders as ≈5% of the bounding
            # box diagonal — large enough to see but not misleading.
            # Explicit multiples (1× – 200×) are used directly.
            _scale_map_def = {
                '1x':   1.0,  '5x':   5.0,  '10x':  10.0,
                '50x': 50.0,  '100x': 100.0, '200x': 200.0,
            }
            _scale_prop = self.get_property('deformation_scale')
            if _scale_prop in _scale_map_def:
                _disp_scale_val = _scale_map_def[_scale_prop]
            else:                                        # 'Auto'
                _bbox_ds = np.ptp(mesh.p, axis=1)
                _clen_ds = max(float(np.max(_bbox_ds)), 1e-9)
                _auto_s  = (0.05 * _clen_ds / _max_disp_val) if _max_disp_val > 1e-9 else 1.0
                _disp_scale_val = float(np.clip(_auto_s, 1.0, 200.0))
            print(f"FEA Solver: Deformation display scale = {_disp_scale_val:.1f}× "
                  f"('{_scale_prop}')")

            # ── Small-strain / geometric nonlinearity warning ─────────────────
            # This solver applies the load to the UNDEFORMED geometry and uses
            # a constant stiffness matrix (geometrically linear).  Results are
            # physically valid only when displacements are small relative to the
            # part's dimensions.  The engineering rule of thumb: max displacement
            # ≤ ~5 % of the bounding-box characteristic length.  Beyond that,
            # follower forces, membrane stiffening, and large-rotation effects
            # can change the answer significantly.
            _bbox_extents = np.ptp(mesh.p, axis=1)          # (3,) range per axis
            _char_len = max(float(np.max(_bbox_extents)), 1e-9)
            _disp_ratio = float(_max_disp_val) / _char_len
            if _disp_ratio > 0.05:
                _geom_nl_msg = (
                    f"FEA Solver: WARNING — Large deflection detected "
                    f"(max |u| = {_max_disp_val:.3f} mm = "
                    f"{_disp_ratio * 100:.1f}% of bounding-box extent "
                    f"{_char_len:.3f} mm).\n"
                    "  This is a small-strain (geometrically linear) solver.\n"
                    "  Results are unreliable when displacements exceed ≈5% of\n"
                    "  the part's characteristic length.  Geometric stiffening,\n"
                    "  membrane action, and follower-force effects are not\n"
                    "  captured.  Consider reducing the applied load or switching\n"
                    "  to a nonlinear (large-deformation) solver."
                )
                print(_geom_nl_msg)
                logger.warning(_geom_nl_msg)
        except np.linalg.LinAlgError as e:
            _sing_msg = (
                f"FEA Solver: Linear solve FAILED — singular stiffness matrix ({e}).\n"
                "  Most likely cause: insufficient boundary conditions (rigid-body motion).\n"
                "  The stiffness matrix K has one or more zero eigenvalues, meaning the\n"
                "  structure is free to translate or rotate without any elastic resistance.\n"
                "  Checks to perform:\n"
                "    1. Ensure at least 6 DOFs are constrained (3 translations + 3 rotations).\n"
                "    2. Constraints must be on non-collinear, non-coplanar nodes.\n"
                "    3. Each global direction (X, Y, Z) must have at least one fixed DOF.\n"
                "  See the RBM warning printed above for specific issues detected."
            )
            print(_sing_msg)
            self.set_error(_sing_msg)
            return None
        except Exception as e:
            print(f"FEA Solver: ERROR during solve: {e}")
            return None

        # 7. Calculate Von Mises Stress
        stress = None
        max_stress_gauss = 0.0
        try:
            # ── FIX #1: Use a standalone SCALAR P1 basis created directly from the
            # mesh — NOT basis.with_element(), which inherits the parent vector
            # element.  Inside a LinearForm for a vector basis, 'v' is a 3-vector;
            # multiplying vm * v returns a 3-component result assembled into the wrong
            # size DOF vector.  A scalar basis makes 'v' a true scalar so vm * v
            # is the correct L2 projection RHS.
            basis_p1 = Basis(mesh, ElementTetP1(), quadrature=basis.quadrature)

            @LinearForm
            def von_mises(v, w):
                # Reconstruct stress tensor from strain
                def epsilon(w):
                    return sym_grad(w)

                E = epsilon(w['u'])
                mu = w['mu']
                lam = w['lam']

                # Components of Strain Tensor E
                E11, E12, E13 = E[0,0], E[0,1], E[0,2]
                E21, E22, E23 = E[1,0], E[1,1], E[1,2]
                E31, E32, E33 = E[2,0], E[2,1], E[2,2]

                trE = E11 + E22 + E33

                # Components of Stress Tensor S
                # S_ij = 2*mu*E_ij + lam*tr(E)*delta_ij
                S11 = 2*mu*E11 + lam*trE
                S22 = 2*mu*E22 + lam*trE
                S33 = 2*mu*E33 + lam*trE
                S12 = 2*mu*E12
                S23 = 2*mu*E23
                S13 = 2*mu*E13

                # Von Mises Stress — scalar return, correct for scalar basis
                vm = np.sqrt(0.5 * ((S11-S22)**2 + (S22-S33)**2 + (S33-S11)**2 + 6*(S12**2 + S23**2 + S13**2)))
                return vm * v

            # Assemble Mass Matrix for scalar P1 basis
            @BilinearForm
            def mass(u, v, w):
                return u * v

            M = mass.assemble(basis_p1)

            # Assemble Load Vector (Projected Stress)
            b = von_mises.assemble(basis_p1,
                u=basis.interpolate(u),
                mu=basis_p1.zeros() + mu_val,
                lam=basis_p1.zeros() + lam_val
            )

            stress = solve(M, b)

            # Ensure stress is positive (numerical errors might make it slightly negative)
            stress = np.abs(stress)
            logger.info(f"Stress Calc Complete. Max Stress (L2 nodal): {np.max(stress):.6e}")

            # ── Gauss-point maximum stress (P0 projection) ────────────────────
            # The L2 projection above is a global low-pass filter: inter-element
            # smoothing smears sharp stress concentrations over neighbouring nodes
            # and under-reports the true peak.  Re-evaluate Von Mises on a
            # P0 (constant-per-element) basis: each entry is the element-average
            # stress evaluated at the actual integration points, with NO
            # inter-element blending.  np.max(stress_p0) is the conservative
            # value to use for safety-factor calculations.
            try:
                basis_p0_vm = basis.with_element(ElementTetP0())
                # ensure quadrature matches
                basis_p0_vm = Basis(mesh, ElementTetP0(), quadrature=basis.quadrature)

                @LinearForm
                def von_mises_p0(v, w):
                    def epsilon_p0(w):
                        return sym_grad(w)
                    Ep  = epsilon_p0(w['u'])
                    m0  = w['mu']
                    l0  = w['lam']
                    trE = Ep[0,0] + Ep[1,1] + Ep[2,2]
                    T11 = 2*m0*Ep[0,0] + l0*trE
                    T22 = 2*m0*Ep[1,1] + l0*trE
                    T33 = 2*m0*Ep[2,2] + l0*trE
                    T12 = 2*m0*Ep[0,1]
                    T23 = 2*m0*Ep[1,2]
                    T13 = 2*m0*Ep[0,2]
                    vm0 = np.sqrt(0.5 * ((T11-T22)**2 + (T22-T33)**2 + (T33-T11)**2
                                        + 6*(T12**2 + T23**2 + T13**2)))
                    return vm0 * v

                @BilinearForm
                def mass_p0(u, v, w):
                    return u * v

                M_p0   = mass_p0.assemble(basis_p0_vm)
                b_p0   = von_mises_p0.assemble(
                    basis_p0_vm,
                    u   = basis.interpolate(u),
                    mu  = basis_p0_vm.zeros() + mu_val,
                    lam = basis_p0_vm.zeros() + lam_val,
                )
                stress_p0        = np.abs(solve(M_p0, b_p0))
                max_stress_gauss = float(np.max(stress_p0))
                logger.info(
                    f"Gauss-point max stress (P0, unsmoothed): "
                    f"{max_stress_gauss:.6e} MPa  "
                    f"| L2-smoothed nodal max: {np.max(stress):.6e} MPa"
                )
            except Exception as _p0_err:
                logger.debug(
                    f"FEA Solver: P0 stress projection failed ({_p0_err}); "
                    "using smoothed nodal max as fallback."
                )
                max_stress_gauss = float(np.max(stress))

        except Exception as e:
            msg = f"FEA Solver: Stress calculation failed: {e}"
            print(msg)
            logger.error(msg)
            stress = None
            max_stress_gauss = 0.0

        # Build perfectly mapped displacement vector for the 3D Viewer (length 3*N_points)
        try:
            n_points = mesh.p.shape[1]
            disp_3n = np.zeros((3, n_points))
            nodal_dofs = basis.nodal_dofs
            
            # Use only DOFs associated with existing mesh vertices for linear visualization
            # nodal_dofs columns correspond to points in mesh.p
            limit = min(nodal_dofs.shape[1], n_points)
            disp_3n[0, :limit] = u[nodal_dofs[0, :limit]]
            disp_3n[1, :limit] = u[nodal_dofs[1, :limit]]
            disp_3n[2, :limit] = u[nodal_dofs[2, :limit]]
            displacement_flat = disp_3n.flatten(order='F')
        except Exception:
            displacement_flat = u # Fallback

        # 8. Debug info for viewer (Show where loads/constraints are)
        # Use the already-resolved 'loads' and 'constraints' lists — avoids
        # calling the non-existent resolve_all_inputs() method.
        debug_loads = []
        # Determine maximum force magnitude for dynamic arrow scaling
        max_f_mag = 1e-9
        for load in load_list:
            if isinstance(load, dict) and 'vector' in load:
                m = float(np.linalg.norm(np.array(load.get('vector', [0,0,0]), dtype=float)))
                if m > max_f_mag:
                    max_f_mag = m

        try:
            for load in load_list:
                if isinstance(load, dict):
                    # Prefer existing viz dict
                    if 'viz' in load and load['viz']:
                        v = dict(load['viz'])
                        # Add relative magnitude if it's a force vector
                        if 'vector' in load:
                            v['vector'] = load['vector']
                            v['relative_mag'] = float(np.linalg.norm(np.array(load['vector'], dtype=float))) / max_f_mag
                        debug_loads.append(v)
                        continue

                    # Fallback for geometry without viz
                    geoms_dbg = load.get('geometries', [load.get('geometry')])
                    geoms_dbg = [g for g in geoms_dbg if g is not None]
                    vec = load.get('vector', [0, 0, 0])
                    for g in geoms_dbg:
                        try:
                            bb = g.BoundingBox()
                            center = [(bb.xmin + bb.xmax)/2,
                                      (bb.ymin + bb.ymax)/2,
                                      (bb.zmin + bb.zmax)/2]
                            v_np = np.array(vec, dtype=float)
                            mag = np.linalg.norm(v_np)
                            if mag > 1e-9:
                                viz_vec = (v_np / mag) * 10
                                debug_loads.append({'start': center, 'vector': viz_vec.tolist(), 'relative_mag': mag / max_f_mag})
                        except Exception:
                            pass
        except Exception:
            pass

        debug_constraints = []
        try:
            for const in constraint_list:
                if isinstance(const, dict):
                    # Always include viz dict if available
                    viz_data = const.get('viz', {}) or {}
                    
                    geoms_dbg = const.get('geometries', [const.get('geometry')])
                    geoms_dbg = [g for g in geoms_dbg if g is not None]
                    for g in geoms_dbg:
                        try:
                            bb = g.BoundingBox()
                            center = [(bb.xmin + bb.xmax)/2,
                                      (bb.ymin + bb.ymax)/2,
                                      (bb.zmin + bb.zmax)/2]
                            c_dict = dict(viz_data)
                            c_dict['pos'] = center
                            if 'bbox' not in c_dict:
                                c_dict['bbox'] = bb
                            debug_constraints.append(c_dict)
                        except Exception:
                            pass
        except Exception:
            pass

        # ── Results summary ─────────────────────────────────────────────────────
        _peak_smooth = float(np.max(stress)) if stress is not None and len(stress) > 0 else 0.0
        print(
            f"\n{'='*52}\n  FEA RESULTS SUMMARY\n"
            f"  Max displacement       : {_max_disp_val:.4e} mm\n"
            f"  Peak stress (L2 smooth): {_peak_smooth:.4e} MPa\n"
            f"  Peak stress (P0 Gauss) : {max_stress_gauss:.4e} MPa\n"
            f"  Deformation scale      : {_disp_scale_val:.1f}\u00d7\n"
            f"{'='*52}\n"
        )

        return {
            'mesh': mesh,
            'displacement': displacement_flat,
            'stress': stress,
            # max_stress_gauss: peak Von Mises at Gauss integration points (P0,
            # no inter-element smoothing).  Use this for safety-factor calcs;
            # np.max(stress) from the L2-smoothed nodal field under-reports by
            # 10-30 % at sharp stress concentrations.
            'max_stress_gauss': max_stress_gauss,
            'deformation_scale': _disp_scale_val,
            'type': 'fea',
            'visualization_mode': self.get_property('visualization'),
            'debug_loads': debug_loads,
            'debug_constraints': debug_constraints
        }

