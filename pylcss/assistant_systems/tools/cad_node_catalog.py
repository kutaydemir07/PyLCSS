# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Assistant-facing metadata for supported Design Studio node types."""

# === CAD Node Types Reference ===
# For use in prompts and documentation
# Property names MUST match the exact create_property names in the node classes.
# This is the supported copilot-facing subset of NODE_CLASS_MAPPING. Legacy
# node types can still deserialize without being offered for new workflows.

CAD_NODE_TYPES = {
    # --- GUI-native parametric geometry ---
    "com.cad.geometry.box": {
        "name": "Box",
        "properties": {
            "length_x": 100.0,
            "width_y": 40.0,
            "height_z": 20.0,
            "centered": True,
        },
        "inputs": ["length", "width", "height"],
        "outputs": ["shape"],
    },
    "com.cad.geometry.cylinder": {
        "name": "Cylinder",
        "properties": {
            "diameter": 20.0,
            "length": 40.0,
            "axis": "Z",
            "centered": True,
        },
        "inputs": ["diameter", "length"],
        "outputs": ["shape"],
    },
    "com.cad.geometry.tube": {
        "name": "Tube",
        "properties": {
            "outer_diameter": 40.0,
            "wall_thickness": 2.0,
            "length": 100.0,
            "axis": "X",
            "centered": True,
        },
        "inputs": ["outer_diameter", "wall_thickness", "length"],
        "outputs": ["shape"],
    },
    "com.cad.geometry.cylindrical_shell": {
        "name": "Cylindrical Shell",
        "properties": {
            "diameter": 70.0,
            "length": 180.0,
            "axis": "X",
            "centered": True,
        },
        "inputs": ["diameter", "length"],
        "outputs": ["shape"],
    },
    "com.cad.geometry.boolean": {
        "name": "Boolean",
        "properties": {"operation": "Union"},
        "inputs": ["base", "tool"],
        "outputs": ["shape"],
    },
    "com.cad.geometry.through_hole": {
        "name": "Through Hole",
        "properties": {
            "diameter": 10.0,
            "axis": "Z",
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
        },
        "inputs": ["shape", "diameter"],
        "outputs": ["shape"],
    },
    "com.cad.geometry.fillet": {
        "name": "Fillet",
        "properties": {"radius": 2.0, "edges": "All"},
        "inputs": ["shape", "radius"],
        "outputs": ["shape"],
    },
    "com.cad.geometry.transform": {
        "name": "Transform",
        "properties": {
            "translate_x": 0.0,
            "translate_y": 0.0,
            "translate_z": 0.0,
            "rotation_axis": "Z",
            "rotation_angle_deg": 0.0,
        },
        "inputs": ["shape"],
        "outputs": ["shape"],
    },
    "com.cad.geometry.linear_pattern": {
        "name": "Linear Pattern",
        "properties": {
            "count": 2,
            "spacing": 20.0,
            "axis": "X",
            "fuse": False,
        },
        "inputs": ["shape"],
        "outputs": ["shape"],
    },
    # --- Expert code-first geometry ---
    "com.cad.code_part": {
        "name": "Code Part / Assembly",
        "description": (
            "Write any geometry as CadQuery code. "
            "Set 'code' to a Python snippet that assigns `result`. "
            "Set 'parameters' to name=value lines (one per line) for parametric dims."
        ),
        "properties": {
            "code": "result = cq.Workplane('XY').box(L, W, H)",
            "parameters": "L=40.0\nW=20.0\nH=8.0",
        },
        "inputs": [],
        "outputs": ["shape"],
    },
    "com.cad.freecad_part": {
        "name": "FreeCAD Part",
        "description": "Interactive FreeCAD-authored part opened by double-click.",
        "properties": {
            "fcstd_filename": "",
            "auto_open_on_double_click": True,
        },
        "inputs": [],
        "outputs": ["shape"],
    },
    # --- Selection ---
    "com.cad.select_face": {
        "name": "Select Geometry",
        "properties": {
            "entity_type": "Face",
            "selector_type": "Direction",
            "direction": ">Z",
        },
        "inputs": ["shape"],
        "outputs": ["workplane"],
    },
    "com.cad.select_face_interactive": {
        "name": "Select Geometry (Interactive)",
        "properties": {
            "entity_type": "Face",
            "picked_face_indices": "",
            "selection_label": "No geometry selected",
        },
        "inputs": ["shape"],
        "outputs": ["workplane"],
    },
    # --- Assembly ---
    "com.cad.assembly": {
        "name": "Assembly",
        "properties": {"assembly_name": "Assembly1", "fuse_parts": False},
        "inputs": ["part_1", "part_2", "part_3", "part_4"],
        "outputs": ["assembly"],
    },
    # --- Analysis ---
    "com.cad.mass_properties": {
        "name": "Mass Properties",
        "properties": {"density": 7.85e-9},
        "inputs": ["shape"],
        "outputs": ["properties"],
    },
    "com.cad.bounding_box": {
        "name": "Bounding Box",
        "properties": {},
        "inputs": ["shape"],
        "outputs": ["dimensions"],
    },
    "com.cad.math_expression": {
        "name": "Math Expression",
        "properties": {"expression": "x + y"},
        "inputs": ["x", "y", "z"],
        "outputs": ["result"],
    },
    "com.cad.measure_distance": {
        "name": "Measure Distance",
        "properties": {},
        "inputs": ["shape_a", "shape_b"],
        "outputs": ["distance"],
    },
    "com.cad.surface_area": {
        "name": "Surface Area",
        "properties": {},
        "inputs": ["shape_in"],
        "outputs": ["area_out"],
    },
    # --- FEA Simulation ---
    "com.cad.sim.material": {
        "name": "Material",
        "properties": {
            "preset": "Steel (Structural)",
            "youngs_modulus": 210000.0,
            "poissons_ratio": 0.3,
            "density": 7.85e-9,
            "thermal_conductivity": 45.0,
        },
        "outputs": ["material"],
    },
    "com.cad.sim.mesh": {
        "name": "Generate Mesh",
        "properties": {
            "mesh_type": "Tet",
            "element_size": 2.0,
            "refinement_size": 0.5,
            "shell_thickness": 1.5,
            "shell_nip": 5,
        },
        "inputs": ["shape", "element_size", "refinement_faces", "refinement_size"],
        "outputs": ["mesh"],
    },
    "com.cad.sim.constraint": {
        "name": "FEA Constraint",
        "properties": {"constraint_type": "Fixed"},
        "inputs": ["mesh", "target_face"],
        "outputs": ["constraints"],
    },
    "com.cad.sim.load": {
        "name": "FEA Load",
        "properties": {
            "load_type": "Force",
            "force_x": 0.0,
            "force_y": -1000.0,
            "force_z": 0.0,
        },
        "inputs": ["mesh", "target_face"],
        "outputs": ["loads"],
    },
    "com.cad.sim.pressure_load": {
        "name": "FEA Pressure Load",
        "properties": {"pressure": 1.0, "direction": "Inward"},
        "inputs": ["mesh", "target_face"],
        "outputs": ["loads"],
    },
    "com.cad.sim.solver": {
        "name": "FEA Solver",
        "properties": {
            "analysis_type": "Linear",
            "deck_only": False,
            "external_solver_path": "",
            "external_work_dir": "",
            "external_timeout_s": 3600.0,
            "visualization": "Von Mises Stress",
            "deformation_scale": "Auto",
        },
        "inputs": ["mesh", "material", "constraints", "loads"],
        "outputs": ["results"],
    },
    "com.cad.sim.topopt_voxel": {
        "name": "Topology Optimization",
        "properties": {
            "workflow_mode": "Guided",
            "design_goal": "Lightweight Stiffness",
            "physics_mode": "Structural",
            "formulation": "Density (SIMP)",
            "manufacturing_process": "None",
            "optimizer": "Auto",
            "load_aggregation": "Weighted Sum",
            "volfrac": 0.4,
            "max_iter": 80,
            "tol": 0.01,
            "density_cutoff": 0.45,
            "structure_mode": "Solid Envelope",
            "structure_cell_size_voxels": 8.0,
            "structure_member_thickness_voxels": 1.0,
            "structure_skin_thickness_voxels": 0.75,
            "lattice_variable_density": True,
            "lattice_min_relative_density": 0.15,
            "lattice_max_relative_density": 0.60,
            "lattice_solid_transition_density": 0.92,
            "validate_after_optimize": False,
            "validation_quality": "Standard",
            "generate_cad_after_optimize": False,
            "cad_reconstruction_method": "Recovered Shape",
        },
        "inputs": [
            "design_domain",
            "material",
            "supports",
            "loads",
            "non_design_regions",
            "joints",
            "load_cases",
            "thermal_sinks",
            "thermal_loads",
        ],
        "outputs": ["result", "recovered_shape"],
    },
    "com.cad.topopt.support": {
        "name": "TopOpt Support",
        "properties": {"support_type": "Fixed"},
        "inputs": ["target_region"],
        "outputs": ["supports"],
    },
    "com.cad.topopt.load": {
        "name": "TopOpt Force",
        "properties": {"force_x": 0.0, "force_y": -1000.0, "force_z": 0.0},
        "inputs": ["target_region"],
        "outputs": ["loads"],
    },
    "com.cad.topopt.non_design_region": {
        "name": "TopOpt Non-Design Region",
        "properties": {"region_type": "Keep Material"},
        "inputs": ["region_shape"],
        "outputs": ["regions"],
    },
    "com.cad.topopt.joint": {
        "name": "TopOpt Joint",
        "properties": {
            "joint_name": "Joint",
            "joint_type": "Spherical",
            "axis": "X",
            "relative_stiffness": 100.0,
        },
        "inputs": ["anchor_a", "anchor_b"],
        "outputs": ["joints"],
    },
    "com.cad.topopt.operating_case": {
        "name": "TopOpt Operating Case",
        "properties": {
            "case_name": "Operating Case 1",
            "weight": 1.0,
            "replace_base_supports": True,
        },
        "inputs": ["supports", "loads", "joints"],
        "outputs": ["load_case"],
    },
    "com.cad.topopt.thermal_sink": {
        "name": "TopOpt Thermal Sink",
        "properties": {},
        "inputs": ["target_region"],
        "outputs": ["thermal_sinks"],
    },
    "com.cad.topopt.heat_load": {
        "name": "TopOpt Heat Load",
        "properties": {
            "case_name": "Thermal Case 1",
            "total_heat": 100.0,
            "weight": 1.0,
        },
        "inputs": ["target_region"],
        "outputs": ["thermal_loads"],
    },
    "com.cad.sim.remesh": {
        "name": "Remesh Surface",
        "properties": {"element_size": 3.0, "mesh_quality": "Medium"},
        "inputs": ["topopt_result"],
        "outputs": ["mesh", "shape"],
    },
    # --- Crash Simulation ---
    "com.cad.sim.crash_material": {
        "name": "Crash Material",
        "properties": {
            "preset": "Steel (Structural A36)",
            "youngs_modulus": 210000.0,
            "poissons_ratio": 0.3,
            "density": 7.85e-9,
            "yield_strength": 250.0,
            "tangent_modulus": 2000.0,
            "failure_strain": 0.20,
            "enable_fracture": True,
            "strain_rate_sensitive": True,
        },
        "outputs": ["crash_material"],
    },
    "com.cad.sim.impact": {
        "name": "Impact Condition",
        "properties": {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "velocity_z": -1.0,
            "application_scope": "Fixed specimen + moving impactor",
            "node_tolerance": 2.0,
            "wall_friction": -1.0,
            "wall_gap_mm": 0.0,
        },
        "inputs": ["impact_face"],
        "outputs": ["impact"],
    },
    "com.cad.sim.crash_solver": {
        "name": "Crash Solver",
        "properties": {
            "end_time": 0.5,
            "n_frames": 30,
            "time_steps": 500,
            "deck_only": False,
            "external_timeout_s": 1800.0,
            "openradioss_starter_path": "",
            "openradioss_engine_path": "",
            "external_work_dir": "",
            "visualization": "Von Mises Stress",
            "disp_scale": 3.0,
            "enable_mass_scaling": False,
            "impactor_mass_kg": 0.0,
        },
        "inputs": ["mesh", "crash_material", "constraints", "impact"],
        "outputs": ["crash_results"],
    },
    "com.cad.sim.radioss_deck": {
        "name": "Run Radioss Deck",
        "properties": {"deck_path": "", "deck_only": False, "timeout_s": 7200.0},
        "outputs": ["crash_results"],
    },
    # --- IO / Values ---
    "com.cad.number": {
        "name": "Number",
        "properties": {"value": 10.0},
        "outputs": ["value"],
    },
    "com.cad.variable": {
        "name": "Variable",
        "properties": {"value": 0.0},
        "outputs": ["value"],
    },
    "com.cad.import_step": {
        "name": "Import STEP",
        "properties": {"filepath": ""},
        "outputs": ["shape_out"],
    },
    "com.cad.import_stl": {
        "name": "Import STL",
        "properties": {"filepath": ""},
        "outputs": ["mesh_out"],
    },
    "com.cad.export_step": {
        "name": "Export STEP",
        "properties": {"filename": "output.step"},
        "inputs": ["shape"],
    },
    "com.cad.export_stl": {
        "name": "Export STL",
        "properties": {"filename": "output.stl", "smoothing": 10},
        "inputs": ["shape"],
    },
}


# === System Modeling Node Types Reference ===
# Port naming: InputNode/OutputNode/IntermediateNode ports are named after var_name.
# CustomBlockNode ports are in_1, in_2, ... / out_1, out_2, ...
# When connecting: use the var_name as port name for I/O/Intermediate nodes.

SYSTEM_NODE_TYPES = {
    "com.pfd.input": {
        "name": "Design Variable",
        "description": "A design variable (input parameter) with bounds for the design space.",
        "properties": {
            "var_name": {
                "type": "string",
                "default": "x",
                "description": "Variable name (also renames the output port)",
            },
            "unit": {
                "type": "string",
                "default": "-",
                "description": "Physical unit (pint-compatible, e.g. 'mm', 'kg', 'N/m^2', or '-' for dimensionless)",
            },
            "min": {
                "type": "string",
                "default": "0.0",
                "description": "Minimum value in design space",
            },
            "max": {
                "type": "string",
                "default": "10.0",
                "description": "Maximum value in design space",
            },
        },
        "inputs": [],
        "outputs": ["<var_name>"],  # Port is named after var_name
    },
    "com.pfd.output": {
        "name": "Quantity of Interest (QoI)",
        "description": "An output quantity with optional requirement bounds and optimization objective.",
        "properties": {
            "var_name": {
                "type": "string",
                "default": "y",
                "description": "Variable name (also renames the input port)",
            },
            "unit": {
                "type": "string",
                "default": "-",
                "description": "Physical unit (pint-compatible)",
            },
            "req_min": {
                "type": "string",
                "default": "-1e9",
                "description": "Requirement lower bound (use -1e9 for unconstrained)",
            },
            "req_max": {
                "type": "string",
                "default": "1e9",
                "description": "Requirement upper bound (use 1e9 for unconstrained)",
            },
            "minimize": {
                "type": "boolean",
                "default": False,
                "description": "Set True to minimize this QoI (objective)",
            },
            "maximize": {
                "type": "boolean",
                "default": False,
                "description": "Set True to maximize this QoI (objective)",
            },
        },
        "inputs": ["<var_name>"],  # Port is named after var_name
        "outputs": [],
    },
    "com.pfd.intermediate": {
        "name": "Intermediate Variable",
        "description": "A pass-through variable connecting functions. Used for chaining black-box outputs to inputs.",
        "properties": {
            "var_name": {
                "type": "string",
                "default": "z",
                "description": "Variable name (renames both ports)",
            },
            "unit": {
                "type": "string",
                "default": "-",
                "description": "Physical unit (pint-compatible)",
            },
        },
        "inputs": ["<var_name>"],
        "outputs": ["<var_name>"],
    },
    "com.pfd.custom_block": {
        "name": "Black Box Function",
        "description": "A Python function block with configurable inputs/outputs. Write code using in_1, in_2, ... as inputs and assign to out_1, out_2, ... as outputs.",
        "properties": {
            "num_inputs": {
                "type": "string",
                "default": "1",
                "description": "Number of input ports (creates in_1, in_2, ...)",
            },
            "num_outputs": {
                "type": "string",
                "default": "1",
                "description": "Number of output ports (creates out_1, out_2, ...)",
            },
            "code_content": {
                "type": "string",
                "default": "# out_1 = in_1 * 2\n",
                "description": "Python code. Use in_1, in_2,... as input variables. Assign results to out_1, out_2,... "
                "Supports numpy (np), math. Example: 'out_1 = in_1**2 + in_2'",
            },
            "use_surrogate": {
                "type": "boolean",
                "default": False,
                "description": "Use trained surrogate model instead of code",
            },
        },
        "inputs": ["in_1", "in_2", "..."],  # Dynamic: in_1 to in_N
        "outputs": ["out_1", "out_2", "..."],  # Dynamic: out_1 to out_N
    },
}
