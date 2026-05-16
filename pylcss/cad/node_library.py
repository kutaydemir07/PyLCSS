# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Central registry for CAD nodes.

PyLCSS is code-first: parametric geometry is authored in a
:class:`~pylcss.cad.nodes.code_part.CadQueryCodeNode` (one readable CadQuery
script per part / assembly), or imported via STEP / STL.  The hand-placed
primitive / sketch / 3-D-op / transform / pattern nodes have been removed.

Maps node identifiers (``com.cad.*``) to their Python classes.
"""

from pylcss.cad.nodes import (
    # Code-based geometry — the primary authoring path.
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
    SolverNode, TopologyOptimizationNode,
    RemeshNode, SizeOptimizationNode, ShapeOptimizationNode,

    # Crash / Impact.
    CrashMaterialNode, ImpactConditionNode, CrashSolverNode, RunRadiossDeckNode,

    # IO — geometry import / export and named parameter scalars.
    ImportStepNode, ImportStlNode,
    ExportStepNode, ExportStlNode,
    NumberNode, VariableNode,
)

# Master mapping of Node ID -> Node Class.  Keep this in lockstep with the
# LibraryPanel toolbar in pylcss/user_interface/cad/cad_widget.py.
NODE_CLASS_MAPPING = {
    # Geometry — code-first.
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
    'com.cad.sim.topopt':         TopologyOptimizationNode,
    'com.cad.sim.remesh':         RemeshNode,
    'com.cad.sim.sizeopt':        SizeOptimizationNode,
    'com.cad.sim.shapeopt':       ShapeOptimizationNode,

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
