# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Central registry for GUI-native CAD and engineering nodes."""

from pylcss.design_studio.nodes import (
    # GUI-native parametric geometry.
    BoxNode, CylinderNode, TubeNode, CylindricalShellNode, BooleanNode, ThroughHoleNode,
    FilletNode, TransformNode, LinearPatternNode,

    # Optional expert geometry.
    CadQueryCodeNode,

    # Interactive geometry (FreeCAD GUI subprocess + BREP round-trip).
    FreeCadPartNode,

    # Face / surface selection (still needed for boundary-condition wiring).
    SelectFaceNode, InteractiveSelectFaceNode,

    # Assembly of multiple shapes (still a useful aggregator).
    AssemblyNode,

    # Analysis utilities.
    MassPropertiesNode, BoundingBoxNode,
    MathExpressionNode, MeasureDistanceNode, SurfaceAreaNode,

    # FEA / Simulation.
    MaterialNode, MeshNode, ConstraintNode, LoadNode, PressureLoadNode,
    SolverNode, TopologyOptVoxelNode,
    RemeshNode,

    # Crash / Impact.
    CrashMaterialNode, ImpactConditionNode, CrashSolverNode, RunRadiossDeckNode,

    # IO — geometry import / export and named parameter scalars.
    ImportStepNode, ImportStlNode,
    ExportStepNode, ExportStlNode,
    NumberNode, VariableNode,
)
from pylcss.design_studio.topology_optimization.integration import (
    TopologyHeatLoadNode,
    TopologyJointNode,
    TopologyLoadNode,
    TopologyOperatingCaseNode,
    TopologySupportNode,
    TopologyThermalSinkNode,
)

# Master mapping of Node ID -> Node Class. The compact public LibraryPanel is a
# curated subset; legacy/expert nodes stay registered so existing studies load.
NODE_CLASS_MAPPING = {
    # Geometry — GUI-native parametric modeling.
    'com.cad.geometry.box': BoxNode,
    'com.cad.geometry.cylinder': CylinderNode,
    'com.cad.geometry.tube': TubeNode,
    'com.cad.geometry.cylindrical_shell': CylindricalShellNode,
    'com.cad.geometry.boolean': BooleanNode,
    'com.cad.geometry.through_hole': ThroughHoleNode,
    'com.cad.geometry.fillet': FilletNode,
    'com.cad.geometry.transform': TransformNode,
    'com.cad.geometry.linear_pattern': LinearPatternNode,

    # Geometry — optional expert path.
    'com.cad.code_part': CadQueryCodeNode,
    # Geometry — interactive (opens FreeCAD GUI on double-click, BREP round-trip).
    'com.cad.freecad_part': FreeCadPartNode,
    'com.cad.import_step': ImportStepNode,
    'com.cad.import_stl':  ImportStlNode,

    # Face / surface selection — needed by FEA / Crash BC wiring.
    'com.cad.select_face':              SelectFaceNode,
    'com.cad.select_face_interactive':  InteractiveSelectFaceNode,

    # Assembly aggregator.
    'com.cad.assembly': AssemblyNode,

    # Analysis utilities.
    'com.cad.mass_properties':  MassPropertiesNode,
    'com.cad.bounding_box':     BoundingBoxNode,
    'com.cad.math_expression':  MathExpressionNode,
    'com.cad.measure_distance': MeasureDistanceNode,
    'com.cad.surface_area':     SurfaceAreaNode,

    # FEA Simulation.
    'com.cad.sim.material':       MaterialNode,
    'com.cad.sim.mesh':           MeshNode,
    'com.cad.sim.constraint':     ConstraintNode,
    'com.cad.sim.load':           LoadNode,
    'com.cad.sim.pressure_load':  PressureLoadNode,
    'com.cad.sim.solver':         SolverNode,
    'com.cad.sim.topopt_voxel':    TopologyOptVoxelNode,
    'com.cad.topopt.support':       TopologySupportNode,
    'com.cad.topopt.load':          TopologyLoadNode,
    'com.cad.topopt.joint':         TopologyJointNode,
    'com.cad.topopt.operating_case': TopologyOperatingCaseNode,
    'com.cad.topopt.thermal_sink':  TopologyThermalSinkNode,
    'com.cad.topopt.heat_load':     TopologyHeatLoadNode,
    'com.cad.sim.remesh':         RemeshNode,

    # Crash / Impact Simulation.
    'com.cad.sim.crash_material': CrashMaterialNode,
    'com.cad.sim.impact':         ImpactConditionNode,
    'com.cad.sim.crash_solver':   CrashSolverNode,
    'com.cad.sim.radioss_deck':   RunRadiossDeckNode,

    # IO — named scalar parameters + geometry / mesh export.
    'com.cad.number':      NumberNode,
    'com.cad.variable':    VariableNode,
    'com.cad.export_step': ExportStepNode,
    'com.cad.export_stl':  ExportStlNode,
}

# Mapping of Class Name -> Node Class (for legacy class-name-based loading).
NODE_NAME_MAPPING = {cls.__name__: cls for cls in NODE_CLASS_MAPPING.values()}
