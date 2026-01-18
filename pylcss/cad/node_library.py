# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Central registry for all CAD nodes.
Maps node identifiers (com.cad.*) to their Python classes.
"""

from pylcss.cad.nodes import (
    # Primitives
    BoxNode, CylinderNode, SphereNode, ConeNode, TorusNode, 
    WedgeNode, PyramidNode,
    
    # Sketching
    SketchNode, SplineNode, EllipseNode, PolylineNode,
    LineSketchNode, ArcSketchNode, ParametricCircleSketchNode,
    ParametricRectangleSketchNode, PolygonSketchNode,
    
    # Operations
    ExtrudeNode, PocketNode, FilletNode, SelectFaceNode,
    CutExtrudeNode, BooleanNode, RevolveNode, CylinderCutNode,
    ChamferNode, ShellNode, SweepNode, LoftNode, HelixNode,
    TwistedExtrudeNode,
    
    # Cutting
    HoleAtCoordinatesNode, MultiHoleNode, RectangularCutNode,
    SlotCutNode, ArrayHolesNode,
    
    # Modifications
    OffsetNode,
    
    # Transforms
    TranslateNode, RotateNode, ScaleNode, MirrorNode,
    
    # Patterns
    LinearPatternNode, CircularPatternNode,
    RadialPatternNode, MirrorPatternNode, GridPatternNode,
    
    # Assembly
    AssemblyNode,
    
    # Analysis
    MassPropertiesNode, BoundingBoxNode,
    
    # Simulation
    MaterialNode, MeshNode, ConstraintNode, LoadNode, PressureLoadNode,
    SolverNode, TopologyOptimizationNode,
    
    # IO
    ExportStepNode, ExportStlNode, NumberNode, VariableNode,
)

# Master mapping of Node ID -> Node Class (58 nodes)
NODE_CLASS_MAPPING = {
    # Primitives (7)
    'com.cad.box': BoxNode,
    'com.cad.cylinder': CylinderNode,
    'com.cad.sphere': SphereNode,
    'com.cad.cone': ConeNode,
    'com.cad.torus': TorusNode,
    'com.cad.wedge': WedgeNode,
    'com.cad.pyramid': PyramidNode,

    # Sketching (8)
    'com.cad.sketch': SketchNode,
    'com.cad.sketch.line': LineSketchNode,
    'com.cad.sketch.arc': ArcSketchNode,
    'com.cad.sketch.circle': ParametricCircleSketchNode,
    'com.cad.sketch.rectangle': ParametricRectangleSketchNode,
    'com.cad.sketch.polygon': PolygonSketchNode,
    'com.cad.spline': SplineNode,
    'com.cad.polyline': PolylineNode,
    'com.cad.ellipse': EllipseNode,

    # 3D Operations (13)
    'com.cad.extrude': ExtrudeNode,
    'com.cad.revolve': RevolveNode,
    'com.cad.sweep': SweepNode,
    'com.cad.loft': LoftNode,
    'com.cad.helix': HelixNode,
    'com.cad.twisted_extrude': TwistedExtrudeNode,
    'com.cad.pocket': PocketNode,
    'com.cad.cut_extrude': CutExtrudeNode,
    'com.cad.cylinder_cut': CylinderCutNode,
    'com.cad.fillet': FilletNode,
    'com.cad.chamfer': ChamferNode,
    'com.cad.shell': ShellNode,
    'com.cad.boolean': BooleanNode,
    'com.cad.select_face': SelectFaceNode,

    # Cutting Operations (5)
    'com.cad.hole_at_coords': HoleAtCoordinatesNode,
    'com.cad.multi_hole': MultiHoleNode,
    'com.cad.rectangular_cut': RectangularCutNode,
    'com.cad.slot_cut': SlotCutNode,
    'com.cad.array_holes': ArrayHolesNode,

    # Modifications (1)
    'com.cad.offset': OffsetNode,

    # Transformations (4)
    'com.cad.translate': TranslateNode,
    'com.cad.rotate': RotateNode,
    'com.cad.scale': ScaleNode,
    'com.cad.mirror': MirrorNode,

    # Patterns (5)
    'com.cad.linear_pattern': LinearPatternNode,
    'com.cad.circular_pattern': CircularPatternNode,
    'com.cad.pattern.radial': RadialPatternNode,
    'com.cad.pattern.mirror': MirrorPatternNode,
    'com.cad.pattern.grid': GridPatternNode,

    # Assembly (1)
    'com.cad.assembly': AssemblyNode,

    # Analysis (2)
    'com.cad.mass_properties': MassPropertiesNode,
    'com.cad.bounding_box': BoundingBoxNode,

    # FEA Simulation (7)
    'com.cad.sim.material': MaterialNode,
    'com.cad.sim.mesh': MeshNode,
    'com.cad.sim.constraint': ConstraintNode,
    'com.cad.sim.load': LoadNode,
    'com.cad.sim.pressure_load': PressureLoadNode,
    'com.cad.sim.solver': SolverNode,
    'com.cad.sim.topopt': TopologyOptimizationNode,

    # IO (4)
    'com.cad.number': NumberNode,
    'com.cad.variable': VariableNode,
    'com.cad.export_step': ExportStepNode,
    'com.cad.export_stl': ExportStlNode,
}

# Mapping of Class Name -> Node Class (for legacy loading)
NODE_NAME_MAPPING = {cls.__name__: cls for cls in NODE_CLASS_MAPPING.values()}
