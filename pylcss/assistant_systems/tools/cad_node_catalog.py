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
        "name": "Code Part",
        "description": "Build CadQuery geometry and assign it to `result`.",
        "properties": {
            "code": "result = cq.Workplane('XY').box(L, W, H)",
            "parameters": "L=40.0\nW=20.0\nH=8.0",
        },
        "inputs": [
            "param_1",
            "param_2",
            "param_3",
            "param_4",
            "param_5",
            "param_6",
        ],
        "outputs": ["shape"],
    },
    "com.cad.freecad_part": {
        "name": "FreeCAD Part",
        "description": "Edit a linked part in FreeCAD.",
        "properties": {
            "fcstd_filename": "",
            "auto_open_on_double_click": True,
        },
        "inputs": [
            "param_1",
            "param_2",
            "param_3",
            "param_4",
            "param_5",
            "param_6",
            "param_7",
            "param_8",
        ],
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
        "outputs": ["selection"],
    },
    "com.cad.select_face_interactive": {
        "name": "Pick Geometry",
        "properties": {
            "entity_type": "Face",
            "picked_face_indices": "",
            "selection_label": "No geometry selected",
        },
        "inputs": ["shape"],
        "outputs": ["selection"],
    },
    # --- Assembly ---
    "com.cad.assembly": {
        "name": "Group Bodies",
        "description": "Group CAD bodies.",
        "properties": {"assembly_name": "Assembly1"},
        "inputs": ["bodies"],
        "outputs": ["shape"],
    },
    # --- Analysis ---
    "com.cad.mass_properties": {
        "name": "Mass Properties",
        "properties": {"density": 7.85e-9},
        "inputs": ["shape"],
        "outputs": ["properties", "mass", "volume"],
    },
    "com.cad.bounding_box": {
        "name": "Bounding Box",
        "properties": {},
        "inputs": ["shape"],
        "outputs": ["dimensions", "length", "width", "height", "volume"],
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
        "inputs": ["shape"],
        "outputs": ["area"],
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
        "inputs": [
            "youngs_modulus",
            "poissons_ratio",
            "density",
            "thermal_conductivity",
        ],
        "outputs": ["material"],
    },
    "com.cad.sim.mesh": {
        "name": "Mesh",
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
    "com.cad.sim.component": {
        "name": "Body",
        "properties": {"component_name": "Body 1"},
        "inputs": ["mesh", "material"],
        "outputs": ["component"],
    },
    "com.cad.sim.constraint": {
        "name": "Support",
        "properties": {"constraint_type": "Fixed"},
        "inputs": ["mesh", "target_face"],
        "outputs": ["constraints"],
    },
    "com.cad.sim.load": {
        "name": "Force",
        "properties": {
            "load_type": "Force",
            "force_x": 0.0,
            "force_y": -1000.0,
            "force_z": 0.0,
        },
        "inputs": [
            "mesh",
            "target_face",
            "force_x",
            "force_y",
            "force_z",
        ],
        "outputs": ["loads"],
    },
    "com.cad.sim.gravity": {
        "name": "Gravity",
        "properties": {
            "load_type": "Gravity",
            "gravity_accel": 9810.0,
            "gravity_direction": "-Y",
        },
        "inputs": ["mesh"],
        "outputs": ["loads"],
    },
    "com.cad.sim.pressure_load": {
        "name": "Pressure",
        "properties": {"pressure": 1.0, "direction": "Inward"},
        "inputs": ["mesh", "target_face", "pressure"],
        "outputs": ["loads"],
    },
    "com.cad.sim.solver": {
        "name": "Static Solver",
        "properties": {
            "analysis_type": "Linear",
            "deck_only": False,
            "external_solver_path": "",
            "external_work_dir": "",
            "external_timeout_s": 3600.0,
        },
        "inputs": ["mesh", "material", "components", "constraints", "loads"],
        "outputs": ["results"],
    },
    "com.cad.sim.topopt_voxel": {
        "name": "Topology Solver",
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
            "density_cutoff": 0.30,
            "exclusion_scope": "All Loads and Supports",
            "exclusion_thickness_mode": "Program Controlled",
            "exclusion_thickness_mm": 2.0,
            "validate_after_optimize": False,
            "validation_quality": "Standard",
            "cad_reconstruction_method": "Auto",
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
        "outputs": ["results", "recovered_shape"],
    },
    "com.cad.sim.lattice_voxel": {
        "name": "Lattice Solver",
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
            "density_cutoff": 0.30,
            "structure_mode": "Gyroid Lattice",
            "lattice_settings_mode": "Guided",
            "lattice_cell_size_mm": 0.0,
            "lattice_member_thickness_mm": 0.0,
            "lattice_skin_thickness_mm": 0.0,
            "lattice_target_relative_density": 0.0,
            "structure_cell_size_voxels": 8.0,
            "structure_member_thickness_voxels": 1.0,
            "structure_skin_thickness_voxels": 0.75,
            "lattice_variable_density": True,
            "lattice_min_relative_density": 0.15,
            "lattice_max_relative_density": 0.60,
            "lattice_solid_transition_density": 0.92,
            "lattice_porosity": "Conservative",
            "exclusion_scope": "All Loads and Supports",
            "exclusion_thickness_mode": "Program Controlled",
            "exclusion_thickness_mm": 2.0,
            "validate_after_optimize": False,
            "validation_quality": "Standard",
            "cad_reconstruction_method": "Auto",
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
        "outputs": ["results", "recovered_shape"],
    },
    "com.cad.topopt.support": {
        "name": "Fixed Support",
        "properties": {"support_type": "Fixed"},
        "inputs": ["target_region"],
        "outputs": ["supports"],
    },
    "com.cad.topopt.load": {
        "name": "Force",
        "properties": {"force_x": 0.0, "force_y": -1000.0, "force_z": 0.0},
        "inputs": ["target_region"],
        "outputs": ["loads"],
    },
    # --- Impact Simulation ---
    "com.cad.sim.impact": {
        "name": "Impact Setup",
        "properties": {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "velocity_z": -1.0,
            "application_scope": "Fixed specimen + moving impactor",
            "impactor_mass_kg": 0.0,
            "node_tolerance": 2.0,
            "wall_friction": -1.0,
            "wall_gap_mm": 0.0,
        },
        "inputs": ["impact_face"],
        "outputs": ["impact"],
    },
    "com.cad.sim.crash_solver": {
        "name": "Impact Solver",
        "properties": {
            "end_time": 0.5,
            "n_frames": 30,
            "time_steps": 500,
            "deck_only": False,
            "external_timeout_s": 1800.0,
            "openradioss_starter_path": "",
            "openradioss_engine_path": "",
            "external_work_dir": "",
            "enable_mass_scaling": False,
        },
        "inputs": ["mesh", "impact_material", "constraints", "impact"],
        "outputs": ["results"],
    },
    # --- IO / Values ---
    "com.cad.number": {
        "name": "Parameter",
        "properties": {"value": 10.0},
        "outputs": ["value"],
    },
    "com.cad.import_step": {
        "name": "Import CAD",
        "properties": {"filepath": ""},
        "inputs": ["file"],
        "outputs": ["shape"],
    },
    "com.cad.import_stl": {
        "name": "Import Mesh",
        "properties": {"filepath": ""},
        "inputs": ["file"],
        "outputs": ["mesh"],
    },
    "com.cad.export_step": {
        "name": "Export STEP",
        "properties": {"filename": "output.step"},
        "inputs": ["shape"],
        "outputs": ["file"],
    },
    "com.cad.export_stl": {
        "name": "Export STL",
        "properties": {"filename": "output.stl", "smoothing": 10},
        "inputs": ["shape"],
        "outputs": ["file"],
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
