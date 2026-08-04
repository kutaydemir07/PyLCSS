# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""CAD property inspector assembled from focused behavior mixins."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from pylcss.design_studio.core.port_schema import describe_port, human_port_label
from pylcss.design_studio.topology_optimization.integration.study_identity import (
    is_density_study_class,
)

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
            min-width: 0px;
            selection-background-color: #4a9eff;
        }
        QLineEdit:focus { border: 1px solid #4a9eff; background: #181b20; }
        QLineEdit:disabled { color: #6b7178; background: #1a1c20; border-color: #2a2d33; }

        QComboBox {
            background: #14161a;
            border: 1px solid #313641;
            border-radius: 6px;
            padding: 3px 6px;
            color: #eef1f5;
            min-height: 18px;
            min-width: 0px;
        }
        QComboBox::drop-down { border: none; width: 14px; }
        QComboBox QAbstractItemView {
            background: #181b20;
            color: #eef1f5;
            selection-background-color: #4a9eff;
            border: 1px solid #313641;
            min-width: 130px;
        }

        QSpinBox, QDoubleSpinBox {
            background: #14161a;
            border: 1px solid #313641;
            border-radius: 6px;
            padding: 3px 6px;
            color: #eef1f5;
            min-height: 18px;
            min-width: 0px;
        }

        /* Check box — explicit indicator so the tick box stays visible.
           Checked = filled accent box, unchecked = empty dark box. */
        QCheckBox { color: #aab0b8; spacing: 7px; background: transparent; min-width: 0px; }
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
            border-radius: 6px; padding: 4px 6px; color: #d6dae0; font-weight: 600;
            min-width: 0px;
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
            "Analysis",
            (
                "analysis_type",
                "end_time",
                "time_steps",
                "enable_mass_scaling",
                "assembly_connection",
                "connection_tolerance",
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
            "Output",
            (
                "visualization",
                "deformation_scale",
                "disp_scale",
                "n_frames",
                "history_samples",
                "acceleration_cfc",
                "force_cfc",
            ),
        ),
        ("Body", ("component_name",)),
        (
            "Solver",
            (
                "analysis_",
                "end_time",
                "time_steps",
                "damping",
                "enable_",
                "contact_",
                "assembly_connection",
                "connection_tolerance",
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
                "exclusion_",
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
                "geometry_source",
                "element_order",
                "element_size",
                "refinement_",
                "shell_",
                "max_size",
                "min_size",
                "order",
                "close_holes",
                "max_surface_faces",
            ),
        ),
        (
            "Advanced",
            (
                "time_step_scale",
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

    # Prefix-based grouping is a reasonable fallback for legacy nodes, but
    # compact native nodes should keep all of their controls together. Without
    # these overrides, Tube.wall_thickness matched Impact.wall_* and Cylinder
    # opened with Length while Diameter was hidden under General.
    _NODE_PROPERTY_SECTIONS = {
        "BoxNode": "Geometry",
        "CylinderNode": "Geometry",
        "TubeNode": "Geometry",
        "CylindricalShellNode": "Geometry",
        "BooleanNode": "Geometry",
        "ThroughHoleNode": "Geometry",
        "FilletNode": "Geometry",
        "TransformNode": "Geometry",
        "LinearPatternNode": "Geometry",
        "AssemblyNode": "Geometry",
        "FreeCadPartNode": "Source",
        "ImportStepNode": "Source",
        "ImportStlNode": "Source",
        "MathExpressionNode": "Value",
        "NumberNode": "Value",
        "VariableNode": "Value",
        "ExportStepNode": "Export",
        "ExportStlNode": "Export",
    }

    # Inspector properties are fallbacks for these ports. Once connected, the
    # graph owns the value and the fallback must not look editable.
    _DRIVEN_PROPERTY_PORTS = {
        "BoxNode": {
            "length_x": "length",
            "width_y": "width",
            "height_z": "height",
        },
        "CylinderNode": {"diameter": "diameter", "length": "length"},
        "TubeNode": {
            "outer_diameter": "outer_diameter",
            "wall_thickness": "wall_thickness",
            "length": "length",
        },
        "CylindricalShellNode": {
            "diameter": "diameter",
            "length": "length",
        },
        "ThroughHoleNode": {"diameter": "diameter"},
        "FilletNode": {"radius": "radius"},
        "MaterialNode": {
            "youngs_modulus": "youngs_modulus",
            "poissons_ratio": "poissons_ratio",
            "density": "density",
            "thermal_conductivity": "thermal_conductivity",
        },
        "MeshNode": {
            "element_size": "element_size",
            "refinement_size": "refinement_size",
        },
        "LoadNode": {
            "force_x": "force_x",
            "force_y": "force_y",
            "force_z": "force_z",
        },
        "PressureLoadNode": {"pressure": "pressure"},
        "ImportStepNode": {"filepath": "file"},
        "ImportStlNode": {"filepath": "file"},
        "CadQueryCodeNode": {
            **{f"param_{index}_value": f"param_{index}" for index in range(1, 7)}
        },
    }

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
            "assembly_name",
            "fuse_parts",
            # Duplicate of the on-canvas value_input text field — kept for
            # NodeGraphQt back-compat but redundant in the inspector.
            "value",
            # Carried on legacy nodes but not honoured by the new solver paths:
            # 'min_safety_factor' is computed by the optimizer, not set by user;
            # 'fixed_faces' was a JSON-string condition list — the new ShapeOpt
            #   only supports SelectFaceNode-driven constraints (logged warning);
            # 'moment_x/y/z' on LoadNode were never wired to either CalculiX
            #   *CLOAD or OpenRadioss; only force_x/y/z are exported.
            "min_safety_factor",
            "fixed_faces",
            "bc_preset",
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
            # Automatic post-topology surface preparation. The engineer
            # chooses the geometry representation and physical mesh size;
            # basic cleanup is a background safeguard, not a study objective.
            "mesh_quality",
            "repair_surface",
            "allow_voxel_fallback",
            # Universal metadata is rendered in its own compact section.
            "description",
            "tags",
            "notes",
            "schema_version",
        }
    )

    _PROPERTY_TOOLTIPS = {
        "analysis_type": "Linear           — fastest, valid for small deflections and elastic only.\n"
        "Nonlinear (Geometric) — large rotations / deflections (NLGEOM).\n"
        "Nonlinear (Plastic)   — plastic yielding too (requires yield_strength on Material).",
        "visualization": "Field used to colour the result mesh in the 3-D viewer.",
        "deformation_scale": "Multiplier on the displayed displacement (visual only). "
        "'Auto' scales so the peak motion is ~5 % of the bounding box.",
        "disp_scale": "Multiplier on the displayed displacement (visual only).",
        "n_frames": "Number of animation frames recorded for playback.",
        "end_time": "Simulation duration. For impact: milliseconds.",
        "time_steps": "Impact only: when mass scaling is enabled, the OpenRadioss target\n"
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
        "assembly_connection": "Automatic Bonded searches component pairs whose bounding boxes "
        "are within the specified tolerance and writes CalculiX tie constraints. "
        "Every detected interface must be reviewed; this is not frictional contact.",
        "connection_tolerance_mm": "Largest gap allowed for bonded interfaces.",
        "component_name": "Name used in solver output.",
        "openradioss_starter_path": "Override the auto-discovered OpenRadioss starter binary.",
        "openradioss_engine_path": "Override the auto-discovered OpenRadioss engine binary.",
        "stress_scale_to_mpa": "Multiplier from anim_to_vtk's native deck stress unit to MPa. "
        "Use 1e6 for tonne-mm-ms decks, 1000 for kg-mm-ms, or 1 when "
        "the converted stress is already MPa.",
        "preset": "Pick a material from the built-in database, or 'Custom' to set fields manually.",
        "youngs_modulus": "Young's modulus E (MPa in the standard mm/t/s unit system).",
        "poissons_ratio": "Poisson's ratio ν (typical 0.27–0.34 for metals).",
        "density": "Mass density ρ (tonne/mm³ — 7.85e-9 for steel).",
        "yield_strength": "Initial yield stress (MPa). Linear studies use it as an allowable; "
        "the plastic law is active only when Analysis Type is Nonlinear (Plastic).",
        "tangent_modulus": "Bilinear hardening slope after yield (MPa).  Set to 0 for perfectly plastic.",
        "failure_strain": "Equivalent plastic strain at element deletion (explicit impact only).",
        "exposed_name": "Name this Number/Variable becomes when the .cad file is called from\n"
        "the system-modeling tab via cad.fea(...)/cad.impact(...)/cad.topopt(...).\n"
        "Empty = not exposed; for VariableNode it defaults to 'variable_name'.",
        # ── SizeOptimization ───────────────────────────────────────────
        "parameters": "JSON list of upstream-shape property names to optimize, e.g.\n"
        '    ["wall_thickness", "fillet_r"]',
        "bounds": "JSON dict of (min, max) per parameter, e.g.\n"
        '    {"wall_thickness": [1.0, 20.0], "fillet_r": [0.5, 5.0]}',
        "optimizer": "SciPy optimizer. COBYLA is the safest gradient-free choice for 1–5\n"
        "parameters when meshing topology may change between evaluations.\n"
        "SLSQP / L-BFGS-B / trust-constr use finite-difference gradients —\n"
        "each step costs (n_params+1) full CAD→mesh→CalculiX evaluations.",
        "element_size": "Target element size [mm] used when re-meshing each evaluation.",
        # ── TopologyOptimization (advanced) ────────────────────────────
        "penal": "SIMP penalization exponent p. Higher gives a sharper 0/1 "
        "split; 3.0 is standard.",
        "yield_stress": "Yield stress σ_y for the PNorm constraint: ||vm||_PNorm ≤ σ_y.",
        "stress_constraint": "Add a P-norm von Mises stress constraint. Auto selects GCMMA.",
        "strain_rate_sensitive": "Use the material preset's internal strain-rate sensitivity for impact runs.",
        "density_cutoff": "Iso value the recovered shape is extracted at; material below it "
        "is removed. 0.3 is the established density-method convention. This is not the "
        "solver's internal minimum density, which only exists to keep the stiffness "
        "matrix non-singular.",
        "max_iter": "Maximum optimization iterations, not a requested final count. "
        "The solve stops earlier only after the convergence tolerance holds "
        "for the configured patience.",
        "tol": "Relative objective-change convergence tolerance. It must hold for "
        "several consecutive iterations before the solve stops.",
        "convergence_patience": "Consecutive iterations that must satisfy the convergence tolerance.",
        "volfrac": "Material budget: target optimized density volume divided by the "
        "design-domain volume. Non-design hardware is reported separately.",
        "rmin": "Density-filter radius, in elements. Sets the smallest member "
            "the design can hold (about twice this) and suppresses checkerboards. "
            "Guided mode keeps this program-controlled.",
        "minimum_member_size_mm": "Thinnest member the design may contain, in "
        "model units — the wall thickness your process can hold. Leave at 0 to "
        "let the program pick it from the grid. Explicit sizes remain fixed "
        "through mesh refinements; an under-resolved request is rejected.",
        "minimum_void_size_mm": "Smallest hole or channel that must remain open, "
        "in model units. Zero uses the minimum member size.",
        "maximum_member_size_mm": "Largest permitted solid-member diameter in "
        "model units. Zero disables the local-density cap.",
        "topology_convergence_enabled": "Repeat the solve on finer grids.",
        "topology_convergence_levels": "Number of grid levels.",
        "structure_mode": "Manufacturing interpretation applied after the density optimization. "
        "It does not change or add optimizer iterations.",
        "lattice_cell_size_mm": "Lattice unit-cell pitch in model units. Unlike "
            "the voxel value, this stays fixed when the analysis grid "
            "changes. 0 falls back to the voxel pitch.",
        "lattice_member_thickness_mm": "Thinnest strut or TPMS wall, in model "
        "units. 0 falls back to the voxel value.",
        "lattice_skin_thickness_mm": "Solid skin over the lattice, in model "
        "units. 0 falls back to the voxel value.",
        "structure_cell_size_voxels": "Lattice repeat size in recovered-grid voxels.",
        "structure_member_thickness_voxels": "Minimum explicit strut diameter or TPMS wall thickness in "
        "recovered-grid voxels. The cell size must be at least four times this value.",
        "structure_skin_thickness_voxels": "Outer conformal skin thickness in recovered-grid voxels.",
        "lattice_variable_density": "Grade the explicit lattice using the optimized density field.",
        "lattice_min_relative_density": "Lowest relative density mapped into a variable-density lattice.",
        "lattice_max_relative_density": "Highest relative density before the solid-transition threshold.",
        "lattice_solid_transition_density": "Density above which the recovered region remains solid instead of lattice.",
        "lattice_porosity": "Used by Honeycomb, which has no cubic homogenized "
        "law, and by retired legacy families with the same limitation. Conservative uses p=1.8; "
        "Balanced and Maximum Porosity are optimistic concept surrogates (p=1.25 "
        "and p=1.0). Every other family ignores this and uses its measured cell "
        "tensor instead. Recovered geometry must always be re-analysed.",
        "optimize_lattice_members": "For BCC and Octet Truss (and retired legacy strut cells), run a "
        "second axial-truss sizing stage across all structural load cases. It checks "
        "allowable stress, Euler buckling, and the displacement limit.",
        "lattice_max_member_thickness_voxels": "Largest strut diameter available to the member-sizing stage, "
        "expressed in source topology voxels.",
        "lattice_member_sizing_iterations": "Maximum damped fully-stressed-design iterations for individual "
        "strut member diameters.",
        "lattice_buckling_length_factor": "Effective-length factor K in the Euler buckling check. Use the "
        "value justified by the cell joint and end-restraint assumptions.",
        "advanced_settings_visible": "Show numerical and manufacturing controls. Leave off for the guided "
        "engineering workflow; guided defaults remain active.",
        "workflow_mode": "Guided selects mesh, filter, optimizer, and stopping defaults from engineering intent. "
        "Expert exposes numerical controls that require a convergence study.",
        "design_goal": "Engineering objective used to select the structural, stress-constrained, thermal, "
        "coupled, or multi-load formulation.",
        "physics_mode": "Physics solved by the topology study. Guided mode derives this from the design goal.",
        "formulation": "Density (SIMP) is robust and supports explicit lattice interpretation. "
        "Level Set evolves a crisp solid boundary and does not create an explicit lattice.",
        "manufacturing_process": "Applies process defaults: build direction for additive, pull-out for moulding, "
        "or a constant section for extrusion. Verify the selected direction on the model.",
        "load_aggregation": "Weighted Sum optimizes average weighted performance. Worst Case protects the least "
        "favourable connected load or operating case.",
        "load_pnorm_p": "Smooth maximum exponent for multiple load cases. Higher approaches a hard worst case "
        "but can make optimization less smooth.",
        "projected_gradient_step": "Expert update size for projected-gradient optimization. Guided Auto chooses "
        "a suitable optimizer for the active constraints.",
        "exclusion_scope": "Automatically keep material at load/support interfaces. All Loads and Supports is "
        "the safe default; None is expert-only because a load can become stranded in void.",
        "exclusion_thickness_mode": "Program Controlled preserves two average voxel lengths. Manual uses the "
        "physical thickness below.",
        "exclusion_thickness_mm": "Physical material-preservation thickness around selected load/support interfaces.",
        "validate_after_optimize": "Re-mesh and independently re-analyse the recovered solid. Strongly recommended "
        "for final designs and required for lattice interpretations.",
        "validation_quality": "Standard performs one independent re-analysis. Mesh Convergence repeats it on a "
        "refined mesh to estimate discretization sensitivity.",
        "print_ready_mesh": "Clean duplicate/degenerate triangles, smooth lightly, and optionally reduce triangles "
        "after surface recovery without filling openings or deleting components. This is geometric cleanup, "
        "not certification of printability.",
        "mesh_decimate_ratio": "Fraction of recovered triangles retained. Keep 1.0 for analysis or small features.",
        "surface_recovery_method": "VTK SDF is the volume-preserving default. Legacy marching cubes is retained "
        "for comparison and older studies.",
        "surface_quality": "Recovery resolution and feature-preserving smoothing policy. Professional uses "
        "the finest supported recovery grid and strictest smoothing pass band.",
        "symmetry": "Mirror the density field about the selected domain centre planes.",
        "extrusion": "Force a constant topology section along this axis.",
        "overhang_build_axis": "Additive build direction used by the self-support projection. Confirm it matches "
        "the intended machine orientation.",
        "overhang_angle_deg": "Minimum self-supporting overhang angle measured "
        "from the build direction.",
        "pull_out_axis": "One-sided mould/tool withdrawal direction. Removes undercuts shadowed along this axis.",
        "max_member_size_voxels": "Maximum solid-member radius in voxels. Zero disables the constraint.",
        "pattern_repeat": "Number of rotationally repeated topology sectors. One disables repetition.",
        "pattern_axis": "Axis of the repeated rotational pattern.",
        "E0": "Solid material modulus used by the voxel optimizer (normally derived from the connected material).",
        "Emin": "Small void stiffness used to keep the voxel system solvable.",
        "nu": "Poisson's ratio used by the structured voxel analysis.",
        "convection_coefficient": "Reduced-order volumetric heat-rejection coefficient in W/(mm³·K). "
        "It does not resolve wetted surface area or fluid flow.",
        "thermal_conductivity": "Solid thermal conductivity used by the voxel thermal solve.",
        "thermal_conductivity_min": "Small void conductivity used to keep the thermal system solvable.",
        "thermal_penal": "Penalization that discourages intermediate thermal density.",
        "structural_weight": "Relative structural-compliance contribution in the coupled structural + thermal objective.",
        "thermal_weight": "Relative steady-conduction contribution in the coupled structural + thermal objective. This does not model thermal strain.",
        "force_x": "Total force component in X (N), distributed over the connected target region.",
        "force_y": "Total force component in Y (N), distributed over the connected target region.",
        "force_z": "Total force component in Z (N), distributed over the connected target region.",
        "moment_x": "Applied moment about global X (N·mm).",
        "moment_y": "Applied moment about global Y (N·mm).",
        "moment_z": "Applied moment about global Z (N·mm).",
        "support_type": "Constraint type applied to the selected topology interface.",
        "region_type": "Keep Material creates a prescribed solid; Keep Void creates an obstacle/exclusion volume.",
        "joint_type": "Idealized kinematic connection between two selected anchor regions.",
        "relative_stiffness": "Penalty stiffness relative to the surrounding topology material. Use the default "
        "unless a joint-stiffness sensitivity study is available.",
        "weight": "Relative importance of this operating or thermal load case.",
        "replace_base_supports": "Use this case's supports instead of the global supports.",
        "replace_global_joints": "Use this case's joints instead of the global joints.",
        "total_heat": "Total heat input for this thermal case (W), distributed over its selected region.",
        "time_step_scale": "OpenRadioss stability scale on the automatic CFL time step. The default 0.9 retains margin.",
        "solver_backend": "Numerical solver used for this analysis. Auto selects the supported installed backend.",
        "run_external_solver": "Run the configured external solver. When disabled, only the solver deck is generated.",
        "history_samples": "Requested number of history samples used for impact quality and response metrics.",
        "force_cfc": "SAE channel-frequency-class filter applied to force history before reporting peaks.",
        "acceleration_cfc": "SAE channel-frequency-class filter applied to acceleration history.",
        "validation_report_path": "Optional physical-validation evidence file associated with this impact model.",
        "validation_status": "Declared physical-validation state. Numerical completion alone does not validate an impact model.",
        "validated_rate_min_per_s": "Lowest strain rate covered by the material validation evidence.",
        "validated_rate_max_per_s": "Highest strain rate covered by the material validation evidence.",
        "damping_alpha": "Legacy mass-proportional damping coefficient. Keep zero unless a calibrated damping model requires it.",
        "damping_beta": "Legacy stiffness-proportional damping coefficient. Keep zero unless a calibrated damping model requires it.",
        "enable_corotation": "Legacy internal-solver geometric nonlinearity switch; OpenRadioss uses its native finite-deformation formulation.",
        "enable_contact": "Enable contact in the selected explicit solver configuration.",
        "contact_stiffness": "Legacy internal contact penalty scale. OpenRadioss uses the generated native contact settings.",
        "contact_thickness": "Legacy internal contact search thickness (mm). OpenRadioss derives contact from the shell/deck definition.",
        "contact_update_interval": "Legacy internal contact-neighbour update interval in explicit time steps.",
        "mass_scaling_threshold": "Legacy mass-scaling activation threshold. OpenRadioss exposes a physical target time step instead.",
        # ── Mesh / Remesh / Impact ─────────────────────────────────────
        "mesh_type": "Tet: linear C3D4 (fast). Tet10: quadratic C3D10 (more accurate, often several times slower). Shell: 3-node reference-surface triangles for explicit impact.",
        "refinement_size": "Local element size at refinement zones [mm].  0 = no local refinement.",
        "shell_thickness": "Physical shell thickness [mm] written to the OpenRadioss shell "
        "property. The CAD geometry must represent the intended reference "
        "surface; PyLCSS does not extract a midsurface from a solid.",
        "shell_nip": "Through-thickness shell integration points. Five is a practical "
        "starting value for elasto-plastic bending; verify convergence.",
        "close_holes": "Explicitly cap open surface boundaries before remeshing. Leave off when openings, "
        "bores, or powder escape paths are intentional; enable only to repair a diagnosed surface defect.",
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
        "entity_type": "Geometric entity to select: face, edge, or vertex.",
        "near_x": "X coordinate [mm] of the point used by Nearest To Point.",
        "near_y": "Y coordinate [mm] of the point used by Nearest To Point.",
        "near_z": "Z coordinate [mm] of the point used by Nearest To Point.",
        "box_min_x": "Minimum X coordinate [mm] of the selection box.",
        "box_max_x": "Maximum X coordinate [mm] of the selection box.",
        "box_min_y": "Minimum Y coordinate [mm] of the selection box.",
        "box_max_y": "Maximum Y coordinate [mm] of the selection box.",
        "box_min_z": "Minimum Z coordinate [mm] of the selection box.",
        "box_max_z": "Maximum Z coordinate [mm] of the selection box.",
        "range_expr": "Boolean coordinate expression used to select entities, for example (x > 0) & (z < 10).",
        "face_index": "Zero-based entity index. Index selection is sensitive to upstream geometry changes.",
        "tag": "CadQuery tag assigned upstream. Prefer tags when a stable semantic selection is available.",
        "picked_face_indices": "Stored entity IDs chosen in the 3-D viewer. Use the Pick control instead of editing this list.",
        "selection_label": "Read-only summary of the entities selected in the 3-D viewer.",
        "length_x": "Box length along X [mm].",
        "width_y": "Box width along Y [mm].",
        "height_z": "Box height along Z [mm].",
        "diameter": "Diameter [mm]. A connected numeric input overrides this value.",
        "outer_diameter": "Tube outside diameter [mm].",
        "wall_thickness": "Tube wall thickness [mm]; must be less than half the outside diameter.",
        "length": "Axial length [mm]. A connected numeric input overrides this value.",
        "axis": "Principal axis used for creation or orientation.",
        "centered": "Center the primitive on the origin instead of starting at zero.",
        "operation": "Union joins material, Cut subtracts Tool from Base, and Intersect keeps their overlap.",
        "center_x": "Feature-center X coordinate [mm].",
        "center_y": "Feature-center Y coordinate [mm].",
        "center_z": "Feature-center Z coordinate [mm].",
        "radius": "Fillet radius [mm]. It must fit the selected edge geometry.",
        "edges": "CadQuery edge selector. All rounds every eligible edge; use a selector to scope the operation.",
        "translate_x": "Translation along X [mm].",
        "translate_y": "Translation along Y [mm].",
        "translate_z": "Translation along Z [mm].",
        "rotation_axis": "Axis about which the shape is rotated.",
        "rotation_angle_deg": "Right-hand-rule rotation angle [degrees].",
        "count": "Total number of pattern instances, including the original.",
        "spacing": "Distance between adjacent pattern instances [mm].",
        "fuse": "Boolean-union pattern instances. Leave off when separate bodies are required.",
        "assembly_name": "Human-readable assembly name used in outputs.",
        "fuse_parts": "Boolean-union the connected parts; leave off to retain a multi-body assembly.",
        "expression": "Safe arithmetic expression over connected x, y, and z inputs.",
        "filepath": "Source file. Relative paths are resolved from the saved project and repository.",
        "filename": "Output file. Relative paths are written beside the saved project.",
        "smoothing": "Taubin smoothing iterations for exported STL meshes. Zero preserves the recovered geometry.",
        "code": "CadQuery Python body. It must assign the final geometry to result.",
        "param_1_name": "Variable name assigned to Code Part input 1.",
        "param_1_value": "Fallback numeric value for Code Part input 1 when unconnected.",
        "param_2_name": "Variable name assigned to Code Part input 2.",
        "param_2_value": "Fallback numeric value for Code Part input 2 when unconnected.",
        "param_3_name": "Variable name assigned to Code Part input 3.",
        "param_3_value": "Fallback numeric value for Code Part input 3 when unconnected.",
        "param_4_name": "Variable name assigned to Code Part input 4.",
        "param_4_value": "Fallback numeric value for Code Part input 4 when unconnected.",
        "param_5_name": "Variable name assigned to Code Part input 5.",
        "param_5_value": "Fallback numeric value for Code Part input 5 when unconnected.",
        "param_6_name": "Variable name assigned to Code Part input 6.",
        "param_6_value": "Fallback numeric value for Code Part input 6 when unconnected.",
        "fcstd_filename": "FreeCAD document used by this node. Relative paths resolve from the saved project.",
        "auto_open_on_double_click": "Open this part in FreeCAD when the node is double-clicked.",
        "value_input": "Numeric value produced by this node.",
        "value": "Fallback numeric value used when the node has no connected input.",
        "variable_name": "Stable parameter name used in formulas and system-modeling mappings.",
        "constraint_type": "Fixed blocks all translations; directional and displacement options constrain only selected DOFs.",
        "displacement_x_enabled": "Apply the entered prescribed displacement in X.",
        "displacement_y_enabled": "Apply the entered prescribed displacement in Y.",
        "displacement_z_enabled": "Apply the entered prescribed displacement in Z.",
        "load_type": "Force distributes the entered total vector; Gravity applies body acceleration.",
        "joint_name": "Human-readable joint identifier used in diagnostics.",
        "case_name": "Human-readable operating-case identifier used in result reporting.",
        "max_surface_faces": "Recovered-surface triangle budget used before volume remeshing.",
        "geometry_source": "Voxels are robust; Surface follows the exported shape.",
        "element_order": "Linear for impact; Quadratic for static FEA.",
        "material_lot_id": "Optional material batch/lot identifier used for impact traceability.",
        "deck_path": "OpenRadioss starter deck (.rad) or keyword deck (.k/.key).",
        "engine_path": "Optional matching OpenRadioss engine control file.",
        "starter_path": "Override the auto-discovered OpenRadioss starter executable.",
        "engine_executable_path": "Override the auto-discovered OpenRadioss engine executable.",
        "work_dir": "Solver working directory. Empty creates an isolated temporary run directory.",
        "timeout_s": "Maximum external-solver wall time [s] before PyLCSS stops waiting.",
        "cad_reconstruction_method": "Solid topology gets a B-rep only where the manufacturing constraint makes one exact: a faired profile spline for an explicit extrusion, or a height-field draw surface for a cast/moulded pull-out direction. Topology, above-cutoff point coverage, volume, surface deviation, and B-rep validity must pass. Any other process keeps its recovered surface; there is no faceted or freeform fallback.",
        "cad_export_filename": "STEP filename for the reconstructed topology geometry.",
        "nelx": "Expert voxel count in X. Guided mode derives resolution automatically.",
        "nely": "Expert voxel count in Y. Guided mode derives resolution automatically.",
        "nelz": "Expert voxel count in Z. Guided mode derives resolution automatically.",
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
        "element_size": "Element Size (mm)",
        "refinement_size": "Local Element Size (mm)",
        "shell_thickness": "Shell Thickness (mm)",
        "shell_nip": "Through-Thickness Points",
        "mesh_type": "Element Formulation",
        "analysis_type": "Analysis Type",
        "assembly_connection": "Assembly Interfaces",
        "connection_tolerance_mm": "Bond Tolerance (mm)",
        "component_name": "Body Name",
        "visualization": "Result Display",
        "deformation_scale": "Deformation Scale",
        "external_solver_path": "CalculiX Application",
        "application_scope": "Scenario Type",
        "velocity_x": "Velocity X (mm/ms)",
        "velocity_y": "Velocity Y (mm/ms)",
        "velocity_z": "Velocity Z (mm/ms)",
        "node_tolerance": "Surface Tolerance (mm)",
        "wall_friction": "Wall Friction",
        "wall_gap_mm": "Wall Gap (mm)",
        "impactor_mass_kg": "Impactor Mass (kg)",
        "end_time": "End Time (ms)",
        "time_steps": "Mass-Scaling Steps",
        "n_frames": "Result Frames",
        "disp_scale": "Deformation Scale",
        "stress_scale_to_mpa": "Native Stress to MPa",
        "enable_mass_scaling": "Mass Scaling",
        "external_timeout_s": "Solver Timeout (s)",
        "openradioss_starter_path": "Starter Path",
        "openradioss_engine_path": "Engine Path",
        "external_work_dir": "Work Directory",
        "deck_only": "Deck Only",
        "youngs_modulus": "Young's Modulus (MPa)",
        "poissons_ratio": "Poisson's Ratio",
        "density": "Density (t/mm³)",
        "yield_strength": "Yield Strength (MPa)",
        "tangent_modulus": "Tangent Modulus (MPa)",
        "failure_strain": "Failure Strain",
        "strain_rate_sensitive": "Strain-Rate Sensitive",
        "exclusion_scope": "Preserve Interfaces",
        "exclusion_thickness_mode": "Preservation Thickness",
        "exclusion_thickness_mm": "Manual Thickness (mm)",
        "geometry_source": "Source",
        "element_order": "Order",
        "close_holes": "Cap Open Boundaries",
        "max_surface_faces": "Surface Triangle Limit",
        "fuse_parts": "Boolean Union Bodies",
        # --- Units that the auto-humanizer cannot infer from the name ---
        "thermal_conductivity": "Thermal Conductivity (mW/mm/K)",
        "force_x": "Force X (N)",
        "force_y": "Force Y (N)",
        "force_z": "Force Z (N)",
        "gravity_accel": "Gravity Acceleration (mm/s²)",
        "gravity_direction": "Gravity Direction",
        "pressure": "Pressure (MPa)",
        "displacement_x": "Prescribed Displacement X (mm)",
        "displacement_y": "Prescribed Displacement Y (mm)",
        "displacement_z": "Prescribed Displacement Z (mm)",
        "displacement_x_enabled": "Prescribe X",
        "displacement_y_enabled": "Prescribe Y",
        "displacement_z_enabled": "Prescribe Z",
        "center_x": "Hole Center X (mm)",
        "center_y": "Hole Center Y (mm)",
        "center_z": "Hole Center Z (mm)",
        "near_x": "Near Point X (mm)",
        "near_y": "Near Point Y (mm)",
        "near_z": "Near Point Z (mm)",
        "box_min_x": "Box Min X (mm)",
        "box_max_x": "Box Max X (mm)",
        "box_min_y": "Box Min Y (mm)",
        "box_max_y": "Box Max Y (mm)",
        "box_min_z": "Box Min Z (mm)",
        "box_max_z": "Box Max Z (mm)",
        "total_heat": "Total Heat Input (mW)",
        "contact_stiffness": "Contact Stiffness (N/mm)",
        "contact_thickness": "Contact Thickness (mm)",
        # --- Selection / study wording ---
        "entity_type": "Entity Type",
        "selector_type": "Selection Rule",
        "range_expr": "Coordinate Range",
        "face_index": "Face Index",
        "condition": "Coordinate Condition",
        "tag": "Selection Name",
        "selection_label": "Selection Name",
        "expression": "Expression",
        "weight": "Load-Case Weight",
        "relative_stiffness": "Joint Stiffness (relative)",
        "acceleration_cfc": "Acceleration Filter (CFC)",
        "force_cfc": "Force Filter (CFC)",
        "history_samples": "Time-History Samples",
        "smoothing": "Smoothing Passes",
        # --- Topology optimization: SIMP internals renamed to engineering
        # language.  These are pyMOTO/SIMP variable names and mean nothing to
        # an engineer who has not implemented the method. ---
        "nelx": "Voxel Grid Cells X",
        "nely": "Voxel Grid Cells Y",
        "nelz": "Voxel Grid Cells Z",
        "volfrac": "Target Volume Fraction",
        "rmin": "Filter Radius (voxels)",
        "minimum_void_size_mm": "Minimum Void Size (mm)",
        "maximum_member_size_mm": "Maximum Member Size (mm)",
        "overhang_angle_deg": "Minimum Overhang Angle (deg)",
        "penal": "Density Penalty (SIMP p)",
        "thermal_penal": "Thermal Density Penalty",
        "E0": "Solid Young's Modulus (MPa)",
        "Emin": "Void Young's Modulus (MPa)",
        "nu": "Poisson's Ratio",
        "tol": "Convergence Tolerance",
        "max_iter": "Maximum Iterations",
        "density_cutoff": "Solid/Void Cutoff Density",
        "yield_stress": "Allowable Stress (MPa)",
        "load_pnorm_p": "Load-Case Aggregation (p-norm)",
        "projected_gradient_step": "Projected-Gradient Step Size",
        "convergence_patience": "Convergence Patience (iterations)",
        "convection_coefficient": "Convection Coefficient (mW/mm²/K)",
        "thermal_conductivity_min": "Void Thermal Conductivity (mW/mm/K)",
        "structural_weight": "Structural Objective Weight",
        "thermal_weight": "Thermal Objective Weight",
        "mesh_decimate_ratio": "Surface Decimation Ratio",
        "minimum_member_size_mm": "Minimum Member Size (mm)",
        "max_member_size_voxels": "Maximum Member Size (voxels)",
        "pattern_repeat": "Pattern Repetitions",
        "validation_displacement_limit_mm": "Displacement Limit (mm)",
        "validation_yield_safety_factor": "Required Yield Safety Factor",
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
        # "step" is deliberately absent: the only STEP-format property
        # (`cad_export_filename`) has an explicit label, whereas
        # `time_step_scale` and `projected_gradient_step` use "step" as an
        # ordinary word and were being rendered as "Time STEP Scale".
        "tpms": "TPMS",
        "vtk": "VTK",
        "x": "X",
        "y": "Y",
        "z": "Z",
    }

    _PATH_PROP_FILTERS = (
        (
            "deck_path",
            "Select OpenRadioss or keyword deck",
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
        self._updating_property = False  # guard against feedback loop
        self._active_pick_connections = None

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
        # taskbar tabs seen when several nodes are added at once.
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

        class _InspectorViewportFilter(QtCore.QObject):
            def __init__(self, host, scroll_widget):
                super().__init__(scroll_widget)
                self._host = host
                self._scroll = scroll_widget

            def eventFilter(self, obj, event):
                if event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
                    vp_w = self._scroll.viewport().width()
                    if vp_w > 0:
                        self._host.setMaximumWidth(vp_w)
                return super().eventFilter(obj, event)

        scroll.viewport().installEventFilter(
            _InspectorViewportFilter(self.props_host, scroll)
        )

    def _add_separator(self):
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        self.layout.addWidget(sep)

    @staticmethod
    def _node_ports(node, direction):
        """Return visible ports in stable node order."""
        getter = (
            getattr(node, "input_ports", None)
            if direction == "input"
            else getattr(node, "output_ports", None)
        )
        if not callable(getter):
            return []
        try:
            ports = getter()
            if isinstance(ports, dict):
                ports = ports.values()
        except Exception:
            return []

        visible = []
        for port in ports:
            try:
                visible_state = getattr(port, "visible", None)
                if callable(visible_state) and not visible_state():
                    continue
                if isinstance(visible_state, bool) and not visible_state:
                    continue
                port_view = getattr(port, "view", None)
                view_visible = getattr(port_view, "isVisible", None)
                if callable(view_visible) and not view_visible():
                    continue
            except Exception:
                # A third-party/legacy port without visibility metadata remains
                # usable and should still appear in the contract summary.
                pass
            visible.append(port)
        return visible

    @staticmethod
    def _node_display_name(node):
        name = getattr(node, "name", "")
        try:
            return str(name() if callable(name) else name)
        except Exception:
            return str(getattr(node, "NODE_NAME", "Node"))

    @staticmethod
    def _connected_port_names(port):
        try:
            connected = list(port.connected_ports())
        except Exception:
            return []
        names = []
        for other in connected:
            try:
                owner = other.node()
                owner_name = PropertiesPanel._node_display_name(owner)
                names.append(
                    f"{owner_name} · {human_port_label(str(other.name()))}"
                )
            except Exception:
                continue
        return names

    def _apply_driven_property_states(self, node):
        """Disable fallback editors while an upstream port drives the value."""
        mapping = self._DRIVEN_PROPERTY_PORTS.get(node.__class__.__name__, {})
        for property_name, port_name in mapping.items():
            widget = self.property_widgets.get(property_name)
            if widget is None:
                continue
            try:
                port = node.get_input(port_name)
                connected = list(port.connected_ports()) if port else []
            except Exception:
                connected = []
            if not connected:
                continue
            widget.setEnabled(False)
            source_names = self._connected_port_names(port)
            source = source_names[0] if source_names else "upstream input"
            existing = str(widget.toolTip() or "").strip()
            driven_note = f"Driven by {source}."
            widget.setToolTip(
                driven_note if not existing else f"{driven_note}\n\n{existing}"
            )

    def _build_node_summary(self, node):
        """Add editable identity, enabled state, and validation status."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("NodeSummary")
        frame.setStyleSheet(
            "#NodeSummary { background:#202329; border:1px solid #30353d; "
            "border-radius:7px; }"
        )
        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setHorizontalSpacing(7)
        layout.setVerticalSpacing(5)

        name_edit = QtWidgets.QLineEdit(self._node_display_name(node))
        name_edit.setToolTip("Node name.")

        def rename():
            value = name_edit.text().strip()
            if not value:
                name_edit.setText(self._node_display_name(node))
                return
            try:
                node.set_name(value)
            except Exception:
                name_edit.setText(self._node_display_name(node))

        name_edit.editingFinished.connect(rename)
        layout.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        layout.addWidget(name_edit, 0, 1, 1, 2)

        enabled = QtWidgets.QCheckBox("Enabled")
        try:
            enabled.setChecked(not bool(node.disabled()))
        except Exception:
            enabled.setChecked(True)

        def set_enabled(checked):
            try:
                node.set_disabled(not bool(checked))
            except Exception:
                return
            QtCore.QTimer.singleShot(
                0,
                lambda n=node: self.display_node(n)
                if self.current_node is n
                else None,
            )

        enabled.toggled.connect(set_enabled)
        layout.addWidget(enabled, 1, 1)

        inputs = self._node_ports(node, "input")
        missing = []
        for port in inputs:
            descriptor = describe_port(node, port, "input")
            if descriptor.required and not self._connected_port_names(port):
                missing.append(descriptor.label)

        error = None
        try:
            error = node.get_error() if node.has_error() else None
        except Exception:
            error = None
        pending = None
        try:
            pending = node.get_pending() if node.has_pending() else None
        except Exception:
            pending = None
        try:
            disabled = bool(node.disabled())
        except Exception:
            disabled = False
        cached_result = getattr(node, "_last_result", None)
        result_producer = node.__class__.__name__ in {
            "SolverNode",
            "CrashSolverNode",
        } or is_density_study_class(node.__class__.__name__)

        if disabled:
            status_text, status_color = "Disabled", "#8b949e"
        elif error:
            status_text, status_color = "Error", "#ff6b6b"
        elif pending:
            status_text, status_color = "Incomplete", "#ffb74d"
        elif missing:
            status_text, status_color = "Incomplete", "#ffb74d"
        elif cached_result is not None and result_producer:
            status_text, status_color = "Solved", "#bb86fc"
        elif cached_result is not None:
            status_text, status_color = "Done", "#66d17a"
        else:
            status_text, status_color = "Ready", "#66d17a"
        status = QtWidgets.QLabel(f"●  {status_text}")
        status.setStyleSheet(f"color:{status_color}; font-weight:700;")
        detail = str(error or pending or "")
        if missing and not error and not pending:
            detail = f"Missing: {', '.join(missing)}"
        if detail:
            status.setToolTip(detail)
        layout.addWidget(QtWidgets.QLabel("Status"), 1, 0)
        layout.addWidget(status, 1, 2, alignment=QtCore.Qt.AlignRight)
        if detail:
            detail_label = QtWidgets.QLabel(detail)
            detail_label.setWordWrap(True)
            detail_label.setStyleSheet(
                f"color:{status_color}; font-size:9px;"
            )
            layout.addWidget(detail_label, 2, 0, 1, 3)

        node_id = getattr(node, "id", "")
        try:
            node_id = node_id() if callable(node_id) else node_id
        except Exception:
            node_id = ""
        type_id = str(getattr(node, "type_", "") or "")
        identity_tip = (
            f"Instance ID: {node_id or '—'}\n"
            f"Node type: {type_id or '—'}"
        )
        name_edit.setToolTip(f"Node name.\n\n{identity_tip}")
        schema_version = node.get_property("schema_version")
        if schema_version is not None:
            name_edit.setToolTip(
                f"{name_edit.toolTip()}\nSchema: v{schema_version}"
            )
        self.props_layout.addWidget(frame)

    def _build_connection_summary(self, node):
        """Show typed input/output contracts and live connections."""
        inputs = self._node_ports(node, "input")
        outputs = self._node_ports(node, "output")
        if not inputs and not outputs:
            return

        group = QtWidgets.QGroupBox("Connections")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(5)
        count_label = QtWidgets.QLabel(
            f"{len(inputs)} input{'s' if len(inputs) != 1 else ''} · "
            f"{len(outputs)} output{'s' if len(outputs) != 1 else ''}"
        )
        count_label.setStyleSheet("color:#7f8792; font-size:9px;")
        layout.addWidget(count_label)

        def add_port_row(port, direction):
            descriptor = describe_port(node, port, direction)
            connected = self._connected_port_names(port)
            row = QtWidgets.QFrame()
            row.setStyleSheet(
                "QFrame { background:#191b20; border:1px solid #2c3037; "
                "border-radius:5px; }"
            )
            row_layout = QtWidgets.QVBoxLayout(row)
            row_layout.setContentsMargins(7, 5, 7, 5)
            row_layout.setSpacing(2)

            title_row = QtWidgets.QHBoxLayout()
            direction_label = "IN" if direction == "input" else "OUT"
            direction_color = "#64b5f6" if direction == "input" else "#4dd0e1"
            title = QtWidgets.QLabel(
                f"<span style='color:{direction_color}; font-weight:700'>"
                f"{direction_label}</span>&nbsp;&nbsp;<b>{descriptor.label}</b>"
            )
            title.setTextFormat(QtCore.Qt.RichText)
            title_row.addWidget(title, 1)
            type_label = QtWidgets.QLabel(descriptor.data_type)
            type_label.setStyleSheet(
                "color:#b8c0ca; background:#292d34; border-radius:4px; "
                "padding:2px 5px; font-size:9px;"
            )
            type_label.setToolTip(descriptor.description)
            title_row.addWidget(type_label)
            row_layout.addLayout(title_row)

            if connected:
                connection_text = "; ".join(connected)
                connection_color = "#66d17a"
            elif descriptor.required:
                connection_text = "Required"
                connection_color = "#ffb74d"
            elif direction == "input":
                connection_text = "Optional"
                connection_color = "#7f8792"
            else:
                connection_text = "Unused"
                connection_color = "#7f8792"
            connection = QtWidgets.QLabel(connection_text)
            connection.setWordWrap(True)
            connection.setStyleSheet(
                f"color:{connection_color}; font-size:9px; border:none;"
            )
            row_layout.addWidget(connection)
            layout.addWidget(row)

        def add_direction_heading(text):
            heading = QtWidgets.QLabel(text)
            heading.setStyleSheet(
                "color:#8f98a5; font-size:9px; font-weight:700; "
                "letter-spacing:0.7px; margin-top:3px;"
            )
            layout.addWidget(heading)

        if inputs:
            add_direction_heading("INPUTS")
            for port in inputs:
                add_port_row(port, "input")
        if outputs:
            add_direction_heading("OUTPUTS")
            for port in outputs:
                add_port_row(port, "output")
        self.props_layout.addWidget(group)

    def _build_notes(self, node):
        """Add one optional human comment without repeating metadata fields."""
        group = QtWidgets.QGroupBox("Comment")
        form = QtWidgets.QFormLayout(group)
        value = node.get_property("notes") or ""
        editor = QtWidgets.QLineEdit(str(value))
        editor.setPlaceholderText("Optional comment")
        editor.editingFinished.connect(
            lambda w=editor: self.update_property(
                "notes",
                w.text().strip(),
            )
        )
        form.addRow(editor)
        self.props_layout.addWidget(group)

    def display_node(self, node):
        """Display specialized inspector for a selected node."""
        self.current_node = node

        # Replace one hidden page rather than removing dozens of child widgets
        # individually.  On Windows, every removed child can transiently enter
        # Qt's top-level-widget list until DeferredDelete is delivered, which
        # was the source of a rapid burst of small white windows/tabs whenever
        # several nodes were rebuilt at once.
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

            if node is None:
                lbl = QtWidgets.QLabel("No Selection")
                lbl.setAlignment(QtCore.Qt.AlignCenter)
                lbl.setStyleSheet("color: #666; font-style: italic;")
                self.props_layout.addWidget(lbl)
                return

            # Route: specialized builders based on node class
            node_class = node.__class__.__name__
            self._build_node_summary(node)
            if node_class == "LatticeInfillNode":
                self._build_lattice_infill_ui(node)
            elif is_density_study_class(node_class):
                self._build_topopt_voxel_ui(node)
            elif node_class in {
                "TopologySupportNode",
                "TopologyLoadNode",
            }:
                self._build_topopt_definition_ui(node)
            elif node_class == "SolverNode":
                self._build_fea_solver_ui(node)
            elif node_class == "CrashSolverNode":
                self._build_crash_solver_ui(node)
            elif node_class == "CadQueryCodeNode":
                self._build_code_part_ui(node)
            elif node_class == "MaterialNode":
                self._build_material_ui(node)
            elif node_class == "InteractiveSelectFaceNode":
                self._build_interactive_select_ui(node)
            elif node_class == "SelectFaceNode":
                self._build_select_face_ui(node)
            elif node_class in ("ConstraintNode", "LoadNode", "PressureLoadNode"):
                self._build_fea_bc_ui(node)
            else:
                self._build_generic_ui(node)
            self._apply_driven_property_states(node)
            # Contracts and documentation remain available but follow the
            # engineering controls so selecting a node exposes its useful
            # settings immediately, even in a narrow inspector.
            if node_class not in {
                "SolverNode",
                "CrashSolverNode",
            } and not is_density_study_class(node_class):
                self._build_connection_summary(node)
            self._build_notes(node)
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
            "LatticeOptVoxelNode": {"Design Intent"},
            "LatticeInfillNode": {"Design Intent"},
            "SolverNode": {"Analysis"},
            "CrashSolverNode": {"Analysis"},
            "MaterialNode": {"Material"},
            "FEAComponentNode": {"Body"},
            "InteractiveSelectFaceNode": {"Geometry Type"},
            "SelectFaceNode": {"Selector", "Parameters"},
            "ConstraintNode": {"Support"},
            "LoadNode": {"Force", "Gravity"},
            "PressureLoadNode": {"Pressure"},
            "ImpactConditionNode": {"Impact"},
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
                QtWidgets.QPushButton,
            ):
                for widget in group.findChildren(widget_type):
                    widget.setMinimumWidth(0)
                    widget.setSizePolicy(
                        QtWidgets.QSizePolicy.Expanding,
                        QtWidgets.QSizePolicy.Fixed,
                    )
                    if isinstance(widget, QtWidgets.QComboBox):
                        widget.setSizeAdjustPolicy(
                            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
                        )
                        widget.setMinimumContentsLength(1)
            for cb in group.findChildren(QtWidgets.QCheckBox):
                cb.setMinimumWidth(0)
                cb.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding if cb.text() else QtWidgets.QSizePolicy.Fixed,
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
