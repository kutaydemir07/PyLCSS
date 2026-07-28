# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""CAD property inspector assembled from focused behavior mixins."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .inspector_boundaries import BoundaryConditionInspectorMixin
from .inspector_controls import ExpressionEdit, InspectorSection
from .inspector_generic import GenericInspectorMixin
from .inspector_selection import SelectionInspectorMixin
from .inspector_studies import StudyInspectorMixin
from .inspector_topology import TopologyInspectorMixin

__all__ = ["ExpressionEdit", "InspectorSection", "PropertiesPanel"]


class PropertiesPanel(
    StudyInspectorMixin,
    TopologyInspectorMixin,
    GenericInspectorMixin,
    SelectionInspectorMixin,
    BoundaryConditionInspectorMixin,
    QtWidgets.QWidget,
):
    """Property inspector for a selected Design Studio node."""

    property_changed = QtCore.Signal(object, str, object, object)

    _INSPECTOR_QSS = """
        #InspectorPanel { background: #1c1e22; }
        QScrollArea { background: transparent; border: none; }
        #qt_scrollarea_viewport { background: transparent; }
        QGroupBox {
            background: #24272d;
            border: 1px solid #2f333a;
            border-radius: 7px;
            margin-top: 13px;
            padding: 9px 7px 7px 7px;
            font-size: 11px;
            font-weight: 600;
            color: #cdd2d9;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 4px;
            color: #6fb3ff;
            font-weight: 700;
        }
        QLabel { color: #aab0b8; font-size: 11px; background: transparent; }
        QLineEdit {
            background: #14161a;
            border: 1px solid #313641;
            border-radius: 6px;
            padding: 4px 6px;
            color: #eef1f5;
            min-height: 18px;
            selection-background-color: #4a9eff;
        }
        QLineEdit:focus { border: 1px solid #4a9eff; background: #181b20; }
        QLineEdit:disabled { color: #6b7178; background: #1a1c20; border-color: #2a2d33; }

        /* Check box — explicit indicator so the tick box stays visible.
           Checked = filled accent box, unchecked = empty dark box. */
        QCheckBox { color: #aab0b8; spacing: 7px; background: transparent; }
        QCheckBox::indicator {
            width: 16px; height: 16px; border-radius: 4px;
            border: 1px solid #3a3f48; background: #14161a;
        }
        QCheckBox::indicator:hover { border: 1px solid #4a9eff; }
        QCheckBox::indicator:checked {
            background: #4a9eff; border: 1px solid #4a9eff;
        }
        QCheckBox::indicator:disabled {
            border: 1px solid #2a2d33; background: #1a1c20;
        }

        QPushButton {
            background: #2a2e35; border: 1px solid #383d46;
            border-radius: 6px; padding: 6px 12px; color: #d6dae0; font-weight: 600;
        }
        QPushButton:hover { background: #323843; border-color: #4a9eff; color: #ffffff; }
        QPushButton:pressed { background: #2a2e35; }
    """

    _PROPERTY_SECTIONS = [
        (
            "Workflow",
            (
                "workflow_mode",
                "design_goal",
                "physics_mode",
                "manufacturing_process",
                "advanced_settings_visible",
            ),
        ),
        (
            "External Solver",
            (
                "external_",
                "openradioss_",
                "calculix_",
                "run_external",
                "deck_only",
                "solver_backend",
                "deck_path",
                "engine_path",
                "engine_executable_path",
                "starter_path",
                "work_dir",
                "timeout_s",
                "stress_scale_to_mpa",
            ),
        ),
        (
            "Visualization",
            ("visualization", "deformation_scale", "disp_scale", "n_frames"),
        ),
        (
            "Solver",
            (
                "analysis_",
                "end_time",
                "time_steps",
                "damping",
                "enable_",
                "contact_",
                "mass_scaling",
                "iterations",
                "max_iter",
                "tol",
                "convergence_",
                "optimizer",
                "load_aggregation",
                "load_pnorm",
                "projected_gradient",
                "move_limit",
                "min_density",
                "penal",
                "filter_radius",
                "update_scheme",
                "filter_type",
                "element_type",
                "shape_recovery",
                "recovery_resolution",
                "smoothing_iterations",
                "density_cutoff",
                "vol_frac",
                "volfrac",
                "rmin",
                "stress_constraint",
                "yield_stress",
                # TopOpt stress constraint. Use exact name for
                # yield_stress so we don't steal Material's
                # yield_strength into Solver.
                "stress_",
            ),
        ),
        (
            "Manufacturing",
            (
                "structure_",
                "lattice_",
                "print_ready_",
                "mesh_decimate_",
                "surface_recovery_",
                "symmetry",
                "extrusion",
                "overhang_",
                "max_member_",
                "pattern_",
                "generate_cad_",
                "cad_reconstruction_",
                "cad_export_",
            ),
        ),
        ("Physics", ("thermal_", "structural_weight")),
        (
            "Material",
            (
                "preset",
                "E",
                "nu",
                "rho",
                "density",
                "poissons_ratio",
                "yield_strength",
                "tangent_modulus",
                "failure_strain",
                "enable_fracture",
                # Crash-only: engineering-facing rate sensitivity.
                "strain_rate_",
            ),
        ),
        (
            "Mesh",
            (
                "mesh_type",
                "mesher",
                "gmsh_",
                "curvature_",
                "element_size",
                "refinement_",
                "shell_",
                "max_size",
                "min_size",
                "order",
            ),
        ),
        (
            "Impact",
            (
                "velocity_",
                "application_scope",
                "node_tolerance",
                "wall_",
                "impactor_mass_kg",
            ),
        ),
        (
            "Geometry",
            (
                "box_",
                "length",
                "width",
                "depth",
                "height",
                "radius",
                "thickness",
                "near_",
                "selector_type",
                "tag",
                "range_expr",
                "direction",
            ),
        ),
        (
            "Load",
            (
                "load_type",
                "force_",
                "vector",
                "magnitude",
                "pressure",
                "gravity_",
                "accel",
            ),
        ),
        ("Constraint", ("constraint_type", "fixed_dofs", "displacement_")),
    ]

    _PROPERTY_HIDE_IF_EMPTY = ("condition", "range_expr", "tag")

    _PROPERTY_HIDE_ALWAYS = frozenset(
        {
            # Solver-backend selectors collapsed to a single backend.
            "solver_backend",
            "run_external_solver",
            # In-house topology-opt element choice (CalculiX uses C3D4 always).
            "element_type",
            # In-house crash-solver tuning knobs (OpenRadioss has its own).  The
            # `time_steps` property remains visible because the OpenRadioss path
            # uses it as the mass-scaling cycle target (end_time / time_steps).
            "damping_alpha",
            "damping_beta",
            "enable_corotation",
            "enable_contact",
            "contact_stiffness",
            "contact_thickness",
            "contact_update_interval",
            "mass_scaling_threshold",
            # Duplicate of the on-canvas value_input text field — kept for
            # NodeGraphQt back-compat but redundant in the inspector.
            "value",
            # Carried on legacy nodes but not honoured by the new solver paths:
            # 'min_safety_factor' is computed by the optimiser, not set by user;
            # 'fixed_faces' was a JSON-string condition list — the new ShapeOpt
            #   only supports SelectFaceNode-driven constraints (logged warning);
            # 'moment_x/y/z' on LoadNode were never wired to either CalculiX
            #   *CLOAD or OpenRadioss; only force_x/y/z are exported.
            "min_safety_factor",
            "fixed_faces",
            "bc_preset",
            "quality_preset",
            "moment_x",
            "moment_y",
            "moment_z",
            # Internal topology-optimization numerical policy. These may exist in
            # old project JSON but are no longer engineering-facing controls.
            "projection",
            "stress_penalty",
            "stress_pnorm_p",
            "heaviside_projection",
            "heaviside_beta_init",
            "heaviside_beta_max",
            "heaviside_beta_step_iters",
            "heaviside_eta",
            "continuation",
            # Internal OpenRadioss/crash numerical policy.
            "hourglass_formulation",
            "hourglass_coefficient",
            # Replaced by the engineering-facing strain_rate_sensitive checkbox.
            "strain_rate_c",
            "strain_rate_p",
        }
    )

    _PROPERTY_TOOLTIPS = {
        "analysis_type": "Linear           — fastest, valid for small deflections and elastic only.\n"
        "Nonlinear (Geometric) — large rotations / deflections (NLGEOM).\n"
        "Nonlinear (Plastic)   — plastic yielding too (requires yield_strength on Material).",
        "analysis_mode": "Same as analysis_type — legacy alias.",
        "visualization": "Field used to colour the result mesh in the 3-D viewer.",
        "deformation_scale": "Multiplier on the displayed displacement (visual only). "
        "'Auto' scales so the peak motion is ~5 % of the bounding box.",
        "disp_scale": "Multiplier on the displayed displacement (visual only).",
        "n_frames": "Number of animation frames recorded for playback.",
        "end_time": "Simulation duration. For crash: milliseconds.",
        "time_steps": "Crash only: when mass scaling is enabled, the OpenRadioss target\n"
        "time step is end_time / time_steps.  Higher = less added mass but\n"
        "a slower run; lower = faster but more artificial inertia.",
        "enable_mass_scaling": "Hold the explicit time step at end_time / time_steps by adding mass to\n"
        "nodes whose CFL bound is below that value (Radioss /DT/NODA/CST).\n"
        "Prevents the 'estimated remaining time' from drifting upward during\n"
        "the run.  Slight artificial inertia is added — turn off for inertia-\n"
        "sensitive impact studies.",
        "deck_only": "Write the solver deck (.inp for CalculiX / .k+.rad for OpenRadioss)\n"
        "but do NOT launch the solver — useful for inspecting the deck.",
        "external_solver_path": "Override the auto-discovered CalculiX ccx binary path.",
        "external_work_dir": "Solver working directory.  Empty = a temp dir is created per run.",
        "external_timeout_s": "Wall-clock timeout for the external solver run (seconds).",
        "openradioss_starter_path": "Override the auto-discovered OpenRadioss starter binary.",
        "openradioss_engine_path": "Override the auto-discovered OpenRadioss engine binary.",
        "stress_scale_to_mpa": "Multiplier from anim_to_vtk's native deck stress unit to MPa. "
        "Use 1e6 for tonne-mm-ms decks, 1000 for kg-mm-ms, or 1 when "
        "the converted stress is already MPa.",
        "preset": "Pick a material from the built-in database, or 'Custom' to set fields manually.",
        "youngs_modulus": "Young's modulus E (MPa in the standard mm/t/s unit system).",
        "poissons_ratio": "Poisson's ratio Î½ (typical 0.27–0.34 for metals).",
        "density": "Mass density ρ (tonne/mm³ — 7.85e-9 for steel).",
        "yield_strength": "Initial yield stress (MPa). Linear studies use it as an allowable; "
        "the plastic law is active only when Analysis Type is Nonlinear (Plastic).",
        "tangent_modulus": "Bilinear hardening slope after yield (MPa).  Set to 0 for perfectly plastic.",
        "failure_strain": "Equivalent plastic strain at element deletion (crash only).",
        "exposed_name": "Name this Number/Variable becomes when the .cad file is called from\n"
        "the system-modeling tab via cad.fea(...)/cad.crash(...)/cad.topopt(...).\n"
        "Empty = not exposed; for VariableNode it defaults to 'variable_name'.",
        # ── SizeOptimization ───────────────────────────────────────────
        "parameters": "JSON list of upstream-shape property names to optimise, e.g.\n"
        '    ["wall_thickness", "fillet_r"]',
        "bounds": "JSON dict of (min, max) per parameter, e.g.\n"
        '    {"wall_thickness": [1.0, 20.0], "fillet_r": [0.5, 5.0]}',
        "optimizer": "SciPy optimiser. COBYLA is the safest gradient-free choice for 1–5\n"
        "parameters when meshing topology may change between evaluations.\n"
        "SLSQP / L-BFGS-B / trust-constr use finite-difference gradients —\n"
        "each step costs (n_params+1) full CAD→mesh→CalculiX evaluations.",
        "gradient_step": "FD step size for the gradient-based optimisers (SLSQP / L-BFGS-B /\n"
        "trust-constr).  Ignored by COBYLA and Nelder-Mead.",
        "max_iterations": "Outer iteration cap on the optimiser.",
        "tolerance": "Convergence tolerance for the optimiser.",
        "element_size": "Target element size [mm] used when re-meshing each evaluation.",
        "max_stress": "Stress constraint [MPa] — used when chosen optimiser supports inequality constraints.",
        "max_volume": "Volume constraint [mm³].  Zero disables the constraint.",
        "objective": "What the optimiser minimises.  For SizeOpt: Min Weight = volume,\n"
        "Min Compliance = u·f, Min Max Stress = peak Von Mises.",
        # ── ShapeOptimization ──────────────────────────────────────────
        "sensitivity_method": "Biological Stress Leveling: heuristic — moves boundary inward at\n"
        "low-stress nodes, outward at high-stress nodes.  Reduces stress\n"
        "concentrations.\n"
        "Adjoint Compliance: rigorous shape derivative −2·W(u).  Drives the\n"
        "boundary toward uniform strain-energy density (minimum compliance).",
        "volume_preservation": "If checked, rescale boundary motion each iteration so total volume\n"
        "stays equal to the initial volume.  Recommended for compliance opt.",
        "step_size": "Per-iteration boundary motion scale (mm).",
        "smoothing_weight": "Laplacian sensitivity smoothing — higher = smoother boundary updates.",
        "max_displacement": "Cap on per-node motion across the whole optimisation [mm].",
        "convergence_tol": "Relative-objective change below which the optimiser stops.",
        # ── TopologyOptimization (advanced) ────────────────────────────
        "penal": "SIMP penalisation p.  Higher = sharper 0/1 split (3.0 is standard).",
        "min_density": "Ersatz minimum density ρ_min.  Prevents singular stiffness.",
        "move_limit": "Max change in ρ_e per iteration (0.2 = ±20 %).",
        "filter_radius": "Density / sensitivity filter radius [mm] — controls minimum feature size.",
        "filter_type": "density: physical density filter (preferred).\nsensitivity: heuristic sensitivity smoothing.",
        "yield_stress": "Yield stress σ_y for the PNorm constraint: ||vm||_PNorm ≤ σ_y.",
        "stress_constraint": "Add a P-norm von Mises stress constraint. Auto selects GCMMA.",
        "strain_rate_sensitive": "Use the material preset's internal strain-rate sensitivity for crash runs.",
        "update_scheme": "MMA: gradient-based, robust.  OC: simpler optimality-criteria update.",
        "simp_bins": "Number of discrete moduli the CalculiX deck quantises into per iteration.",
        "shape_recovery": "Run marching-cubes shape recovery with Taubin smoothing on the final density field.",
        "recovery_resolution": "Marching-cubes voxel-grid resolution for recovered shape extraction.",
        "smoothing_iterations": "Gaussian smoothing passes on the recovered shape.",
        "density_cutoff": "Density threshold below which material is removed in the recovered shape.",
        "vol_frac": "Target volume fraction (kept / total) — the SIMP constraint.",
        "symmetry_x": "Mirror-plane X coordinate [mm].  None = no X-symmetry.",
        "symmetry_y": "Mirror-plane Y coordinate [mm].  None = no Y-symmetry.",
        "symmetry_z": "Mirror-plane Z coordinate [mm].  None = no Z-symmetry.",
        "iterations": "Outer iteration cap on the SIMP loop.",
        "max_iter": "Maximum optimization iterations, not a requested final count. "
        "The solve stops earlier only after the convergence tolerance holds "
        "for the configured patience.",
        "tol": "Relative objective-change convergence tolerance. It must hold for "
        "several consecutive iterations before the solve stops.",
        "convergence_patience": "Consecutive iterations that must satisfy the convergence tolerance.",
        "volfrac": "Material budget: target optimized density volume divided by the "
        "design-domain volume. Non-design hardware is reported separately.",
        "rmin": "Density-filter radius in voxels; controls the numerical minimum "
        "feature scale and suppresses checkerboards.",
        "structure_mode": "Manufacturing interpretation applied after the density optimization. "
        "It does not change or add optimizer iterations.",
        "structure_cell_size_voxels": "Lattice/rib repeat size in recovered-grid voxels.",
        "structure_member_thickness_voxels": "Explicit strut or wall half-thickness in recovered-grid voxels. "
        "The cell size must be at least four times this value.",
        "structure_skin_thickness_voxels": "Outer conformal skin thickness in recovered-grid voxels.",
        "lattice_variable_density": "Grade the explicit lattice using the optimized density field.",
        "lattice_min_relative_density": "Lowest relative density mapped into a variable-density lattice.",
        "lattice_max_relative_density": "Highest relative density before the solid-transition threshold.",
        "lattice_solid_transition_density": "Density above which the recovered region remains solid instead of lattice.",
        "advanced_settings_visible": "Show numerical and manufacturing controls. Leave off for the guided "
        "engineering workflow; guided defaults remain active.",
        # ── Mesh / Remesh / Impact ─────────────────────────────────────
        "mesh_type": "Tet: linear C3D4 (fast). Tet10: quadratic C3D10 (more accurate, often several times slower). Shell: 3-node reference-surface triangles for crash.",
        "mesher": "Gmsh is the recommended CAD-conforming mesher and supports face, "
        "edge, and vertex refinement. Netgen is retained as a fallback and "
        "supports face refinement only.",
        "gmsh_algorithm": "HXT: parallel Delaunay, usually fastest for large tetrahedral meshes.\n"
        "Delaunay: the most robust Gmsh 3-D algorithm.\n"
        "Frontal (Netgen): often good quality, but does not support the "
        "background size field used for local refinement.",
        "curvature_elements": "Gmsh target elements per full 2π of curvature. Higher values "
        "resolve curved edges more finely; 0 disables curvature sizing.",
        "refinement_size": "Local element size at refinement zones [mm].  0 = no local refinement.",
        "shell_thickness": "Physical shell thickness [mm] written to the OpenRadioss shell "
        "property. The CAD geometry must represent the intended reference "
        "surface; PyLCSS does not extract a midsurface from a solid.",
        "shell_nip": "Through-thickness shell integration points. Five is a practical "
        "starting value for elasto-plastic bending; verify convergence.",
        "close_holes": "Cap small holes during remesh — helps surface watertightness.",
        "repair_surface": "Run topological repair before remeshing.",
        "mesh_quality": "Target mesh-quality factor for remeshing.",
        "velocity_x": "Initial impactor velocity along X [mm/ms = m/s].",
        "velocity_y": "Initial impactor velocity along Y [mm/ms = m/s].",
        "velocity_z": "Initial impactor velocity along Z [mm/ms = m/s].",
        "application_scope": "Fixed specimen + moving impactor: selected face is hit by a moving\n"
        "finite-mass impactor/wall; connected constraints stay active.\n"
        "Moving body + fixed wall: the mesh receives initial velocity and\n"
        "hits a generated fixed wall; connected constraints are ignored.\n"
        "Prescribed moving wall: selected face is driven by a massless platen\n"
        "with imposed velocity, useful for controlled crush.",
        "node_tolerance": "Distance [mm] within which a mesh node is treated as belonging to the impact face.",
        "wall_friction": "Rigid-wall Coulomb friction.  Use -1 for the scenario default\n"
        "(0.0 for fixed barrier, 0.08 for moving platen/impactor).",
        "wall_gap_mm": "Initial clearance from wall to selected/leading face [mm].\n"
        "Use 0 for automatic clearance based on model size.",
        "impactor_mass_kg": "Optional sled/impactor mass [kg].  In Fixed specimen + moving\n"
        "impactor this is the moving rigid wall mass.  In Moving body +\n"
        "fixed wall it is lumped onto the projectile trailing edge.\n"
        "A zero mass moving wall is prescribed velocity, not inertial impact.",
        "enable_fracture": "Delete elements whose plastic strain exceeds failure_strain.",
        # ── Constraint / Load extras ───────────────────────────────────
        "condition": "Optional NumPy boolean expression over the mesh-node coordinates\n"
        "x, y, z (mm).  Used only when no SelectFace node is connected.\n"
        "Example:  z < 0.01   or   (x < 1) & (y > 9).",
        "displacement_x": "Prescribed X-displacement [mm] (only when constraint_type = Displacement).",
        "displacement_y": "Prescribed Y-displacement [mm] (only when constraint_type = Displacement).",
        "displacement_z": "Prescribed Z-displacement [mm] (only when constraint_type = Displacement).",
        "gravity_accel": "Gravity magnitude [mm/s²].  9810 = standard Earth gravity.",
        "gravity_direction": "Sign and axis of the gravity vector.",
        "pressure": "Surface pressure [N/mm² = MPa].  Positive = outward, negative = inward.",
        # ── SelectFace ─────────────────────────────────────────────────
        "selector_type": "How this node picks faces (see the dropdown's own tooltip).",
        "direction": "Outward-normal direction the selected face(s) must point in.",
    }

    _PROPERTY_LABELS = {
        "length_x": "Length X (mm)",
        "width_y": "Width Y (mm)",
        "height_z": "Height Z (mm)",
        "length": "Length (mm)",
        "width": "Width (mm)",
        "height": "Height (mm)",
        "diameter": "Diameter (mm)",
        "outer_diameter": "Outer Diameter (mm)",
        "wall_thickness": "Wall Thickness (mm)",
        "centered": "Center on Origin",
        "axis": "Axis",
        "operation": "Operation",
        "edge_selection": "Edges",
        "radius": "Radius (mm)",
        "count": "Instance Count",
        "spacing": "Spacing (mm)",
        "fuse": "Fuse Instances",
        "translate_x": "Move X (mm)",
        "translate_y": "Move Y (mm)",
        "translate_z": "Move Z (mm)",
        "rotate_x": "Rotate X (deg)",
        "rotate_y": "Rotate Y (deg)",
        "rotate_z": "Rotate Z (deg)",
        "exposed_name": "Exposed Parameter Name",
        "variable_name": "Parameter Name",
        "value_input": "Value",
        "filepath": "Source File",
        "fcstd_filename": "FreeCAD File",
        "cad_export_filename": "STEP Export File",
        "cad_reconstruction_method": "CAD Recovery Method",
        "generate_cad_after_optimize": "Build CAD After Solve",
        "element_size": "Element Size (mm)",
        "refinement_size": "Local Element Size (mm)",
        "shell_thickness": "Shell Thickness (mm)",
        "shell_nip": "Through-Thickness Points",
        "mesh_type": "Element Formulation",
        "analysis_type": "Analysis Type",
        "visualization": "Result Display",
        "deformation_scale": "Deformation Scale",
        "external_solver_path": "CalculiX Application",
        "application_scope": "Crash Scenario",
        "velocity_x": "Velocity X (mm/ms)",
        "velocity_y": "Velocity Y (mm/ms)",
        "velocity_z": "Velocity Z (mm/ms)",
        "node_tolerance": "Node Tolerance (mm)",
        "wall_friction": "Wall Friction",
        "wall_gap_mm": "Wall Gap (mm)",
        "impactor_mass_kg": "Impactor / Sled Mass (kg)",
        "end_time": "End Time (ms)",
        "time_steps": "Mass-Scaling Steps",
        "n_frames": "Animation Frames",
        "disp_scale": "Display Scale",
        "stress_scale_to_mpa": "Native Stress to MPa",
        "enable_mass_scaling": "Mass Scaling",
        "external_timeout_s": "Solver Timeout (s)",
        "openradioss_starter_path": "Starter Path",
        "openradioss_engine_path": "Engine Path",
        "external_work_dir": "Work Directory",
        "deck_only": "Deck Only",
        "youngs_modulus": "Young's Modulus (MPa)",
        "poissons_ratio": "Poisson's Ratio",
        "density": "Density (t/mm^3)",
        "yield_strength": "Yield Strength (MPa)",
        "tangent_modulus": "Tangent Modulus (MPa)",
        "failure_strain": "Failure Strain",
        "strain_rate_sensitive": "Strain-Rate Sensitive",
    }

    _PROPERTY_ACRONYMS = {
        "am": "AM",
        "cad": "CAD",
        "cfrp": "CFRP",
        "fea": "FEA",
        "gcmma": "GCMMA",
        "id": "ID",
        "mma": "MMA",
        "mpa": "MPa",
        "oc": "OC",
        "stl": "STL",
        "step": "STEP",
        "tpms": "TPMS",
        "vtk": "VTK",
        "x": "X",
        "y": "Y",
        "z": "Z",
    }

    _PATH_PROP_FILTERS = (
        (
            "deck_path",
            "Select OpenRadioss / LS-DYNA deck",
            "Solver decks (*.k *.key *.rad *.inp);;All files (*)",
        ),
        (
            "engine_path",
            "Select OpenRadioss engine file",
            "Radioss engine files (*.rad *_0001.rad);;All files (*)",
        ),
        (
            "engine_executable_path",
            "Select OpenRadioss engine binary",
            "Executables (*.exe);;All files (*)",
        ),
        (
            "starter_path",
            "Select OpenRadioss starter binary",
            "Executables (*.exe);;All files (*)",
        ),
        (
            "openradioss_starter_path",
            "Select OpenRadioss starter binary",
            "Executables (*.exe);;All files (*)",
        ),
        (
            "openradioss_engine_path",
            "Select OpenRadioss engine binary",
            "Executables (*.exe);;All files (*)",
        ),
        (
            "external_solver_path",
            "Select CalculiX `ccx` binary",
            "Executables (*.exe);;All files (*)",
        ),
        (
            "filepath",
            "Select CAD file",
            "CAD files (*.step *.stp *.iges *.igs *.brep *.stl *.obj);;All files (*)",
        ),
    )

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("InspectorPanel")
        self.setStyleSheet(self._INSPECTOR_QSS)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(8, 7, 8, 7)
        self.layout.setSpacing(5)
        self.current_node = None
        self.property_widgets = {}
        self._study_status_labels = {}
        self._updating_property = False  # guard against feedback loop
        self._active_pick_connections = None

        # Title
        title = QtWidgets.QLabel("Properties")
        title.setStyleSheet("font-weight: 800; font-size: 13px; color: #E0E0E0;")
        title.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.layout.addWidget(title)

        # Separator
        self._add_separator()

        # Properties area (scrollable)
        scroll = QtWidgets.QScrollArea()
        self.scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Keep a permanent widget inside the scroll area.  Inspector pages are
        # swapped inside this host and are never detached from a parent.  This
        # matters on Windows: a temporarily parentless QWidget can be promoted
        # to a native top-level window, producing the burst of small white
        # taskbar tabs seen when Study Definition adds several nodes.
        self.props_host = QtWidgets.QWidget()
        self.props_host_layout = QtWidgets.QVBoxLayout(self.props_host)
        self.props_host_layout.setContentsMargins(0, 0, 0, 0)
        self.props_host_layout.setSpacing(0)

        self.props_widget = QtWidgets.QWidget(self.props_host)
        self.props_widget.setMinimumWidth(0)
        self.props_layout = QtWidgets.QVBoxLayout(self.props_widget)
        self.props_layout.setContentsMargins(2, 2, 2, 2)
        self.props_layout.setSpacing(5)
        self.props_layout.setAlignment(QtCore.Qt.AlignTop)
        self.props_host_layout.addWidget(self.props_widget)
        scroll.setWidget(self.props_host)
        self.layout.addWidget(scroll)

    def _add_separator(self):
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        self.layout.addWidget(sep)

    def display_node(self, node):
        """Display specialized inspector for a selected node."""
        self.current_node = node

        # Replace one hidden page rather than removing dozens of child widgets
        # individually.  On Windows, every removed child can transiently enter
        # Qt's top-level-widget list until DeferredDelete is delivered, which
        # was the source of the rapid burst of small white windows/tabs after a
        # Study Definition quick-add.
        old_page = self.props_widget
        new_page = QtWidgets.QWidget(self.props_host)
        new_page.setMinimumWidth(0)
        new_layout = QtWidgets.QVBoxLayout(new_page)
        new_layout.setContentsMargins(2, 2, 2, 2)
        new_layout.setSpacing(5)
        new_layout.setAlignment(QtCore.Qt.AlignTop)

        # replaceWidget() preserves the parent hierarchy throughout the swap:
        # neither the old page nor any of its controls can become a top-level
        # native window between calls.
        old_page.hide()
        self.props_host_layout.replaceWidget(old_page, new_page)
        old_page.deleteLater()
        self.props_widget = new_page
        self.props_layout = new_layout

        self.props_widget.setUpdatesEnabled(False)
        try:
            self.property_widgets.clear()
            self._study_status_labels.clear()

            if node is None:
                lbl = QtWidgets.QLabel("No Selection")
                lbl.setAlignment(QtCore.Qt.AlignCenter)
                lbl.setStyleSheet("color: #666; font-style: italic;")
                self.props_layout.addWidget(lbl)
                return

            # Node Header
            node_name = node.name() if callable(node.name) else node.name
            node_name = (
                str(node_name)
                .replace("Topology Opt (Voxel)", "Topology Optimization")
                .replace("TopOpt", "Topology")
            )
            header = QtWidgets.QLabel(f"{node_name}")
            header.setWordWrap(True)
            header.setStyleSheet(
                "font-weight: 700; font-size: 14px; color:#eef1f5; "
                "margin:2px 1px 3px 1px;"
            )
            header.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            self.props_layout.addWidget(header)

            node_type = str(getattr(node, "NODE_NAME", "") or node.__class__.__name__)
            node_type = node_type.removesuffix("Node").replace("_", " ").strip()
            if node_type and node_type.casefold() != str(node_name).casefold():
                sub_header = QtWidgets.QLabel(node_type)
                sub_header.setStyleSheet(
                    "color:#7f8792; font-size:10px; margin:0 1px 5px 1px;"
                )
                sub_header.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                self.props_layout.addWidget(sub_header)

            # Route: specialized builders based on node class
            node_class = node.__class__.__name__
            if node_class == "TopologyOptVoxelNode":
                self._build_topopt_voxel_ui(node)
            elif node_class == "SolverNode":
                self._build_fea_solver_ui(node)
            elif node_class == "CrashSolverNode":
                self._build_crash_solver_ui(node)
            elif node_class == "CadQueryCodeNode":
                self._build_code_part_ui(node)
            elif node_class in ("MaterialNode", "CrashMaterialNode"):
                self._build_material_ui(node)
            elif node_class == "InteractiveSelectFaceNode":
                self._build_interactive_select_ui(node)
            elif node_class == "SelectFaceNode":
                self._build_select_face_ui(node)
            elif node_class in ("ConstraintNode", "LoadNode", "PressureLoadNode"):
                self._build_fea_bc_ui(node)
            else:
                self._build_generic_ui(node)
            self._compact_inspector_content(node_class)
        finally:
            self.props_widget.setUpdatesEnabled(True)

    def _compact_inspector_content(self, node_class):
        """Wrap direct property groups and make narrow forms readable."""
        groups = []
        for index in range(self.props_layout.count()):
            item = self.props_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if isinstance(widget, QtWidgets.QGroupBox):
                groups.append((index, widget, str(widget.title() or "Properties")))

        if not groups:
            return

        preferred_open = {
            "TopologyOptVoxelNode": {"Design Intent"},
            "SolverNode": {"Study Definition", "Solver"},
            "CrashSolverNode": {"Study Definition", "Solver"},
            "MaterialNode": {"Material Definition"},
            "CrashMaterialNode": {"Material Definition"},
            "InteractiveSelectFaceNode": {"Face Selection"},
            "SelectFaceNode": {"Selector", "Parameters"},
            "ConstraintNode": {"Boundary Condition"},
            "LoadNode": {"Load"},
            "PressureLoadNode": {"Pressure"},
        }.get(node_class, set())

        for ordinal, (index, group, title) in enumerate(reversed(groups)):
            # Every QFormLayout gets a narrow-panel policy. Long labels move
            # above their editor instead of squeezing the editor to a sliver.
            for form in group.findChildren(QtWidgets.QFormLayout):
                form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
                form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                form.setHorizontalSpacing(7)
                form.setVerticalSpacing(5)
                form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            for widget_type in (
                QtWidgets.QComboBox,
                QtWidgets.QLineEdit,
                QtWidgets.QSpinBox,
                QtWidgets.QDoubleSpinBox,
            ):
                for widget in group.findChildren(widget_type):
                    widget.setMinimumWidth(0)
                    widget.setSizePolicy(
                        QtWidgets.QSizePolicy.Expanding,
                        QtWidgets.QSizePolicy.Fixed,
                    )
            for label in group.findChildren(QtWidgets.QLabel):
                if label.wordWrap():
                    label.setMinimumWidth(0)
                    label.setSizePolicy(
                        QtWidgets.QSizePolicy.Ignored,
                        QtWidgets.QSizePolicy.Preferred,
                    )

            item = self.props_layout.takeAt(index)
            if item is None or item.widget() is not group:
                continue
            expanded = (
                title in preferred_open if preferred_open else index == groups[0][0]
            )
            wrapper = InspectorSection(title, group, expanded, self.props_widget)
            self.props_layout.insertWidget(index, wrapper)
