# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.


"""
Central registry for all CAD nodes.
Maps node identifiers (com.cad.*) to their Python classes.
"""

from pylcss.cad.nodes import (
    BoxNode, CylinderNode, SphereNode,
    ExtrudeNode, PocketNode, FilletNode,
    NumberNode, VariableNode, ExportStepNode, ExportStlNode, 
    SelectFaceNode, CutExtrudeNode, BooleanNode, 
    RevolveNode, CylinderCutNode
)

from pylcss.cad.advanced_nodes import (
    SketchNode, RectangleSketchNode, CircleSketchNode,
    DistanceConstraintNode, AngleConstraintNode, CoincidentConstraintNode,
    AssemblyNode, MateNode,
    MassPropertiesNode, StressAnalysisNode, BoundingBoxNode,
    ForceNode, FixedSupportNode, SimulationRunNode,
    DrawingNode, PropertyTableNode, ReportGeneratorNode
)

from pylcss.cad.parametric_nodes import (
    ConeNode, TorusNode, WedgeNode, PyramidNode,
    DatumPlaneNode,
    LineNode, ArcNode, PolygonNode, SplineNode, EllipseNode,
    SweepNode, LoftNode, HelixNode,
    ChamferNode, ShellNode, OffsetNode, DraftNode,
    TranslateNode, RotateNode, ScaleNode, MirrorNode,
    LinearPatternNode, CircularPatternNode,
    VolumeNode, SurfaceAreaNode, CenterOfMassNode,
    TextNode, ThreadNode, SplitNode, MeasureDistanceNode
)

from pylcss.cad.enhanced_nodes import (
    CoordinateBoxNode, HoleAtCoordinatesNode, MultiHoleNode,
    RectangularCutNode, SlotCutNode, ArrayHolesNode
)

from pylcss.cad.nodes_impl.simulation import (
    MaterialNode, MeshNode, ConstraintNode, LoadNode, PressureLoadNode, SolverNode, TopologyOptimizationNode
)

from pylcss.cad.nodes_impl.sketch import (
    LineSketchNode, ArcSketchNode, ParametricCircleSketchNode, ParametricRectangleSketchNode, PolygonSketchNode
)

from pylcss.cad.nodes_impl.patterns import (
    RadialPatternNode, MirrorPatternNode, GridPatternNode
)

# Master mapping of Node ID -> Node Class
NODE_CLASS_MAPPING = {
    # Primitives
    'com.cad.box': BoxNode,
    'com.cad.cylinder': CylinderNode,
    'com.cad.sphere': SphereNode,
    'com.cad.cone': ConeNode,
    'com.cad.torus': TorusNode,
    'com.cad.wedge': WedgeNode,
    'com.cad.pyramid': PyramidNode,
    'com.cad.coordinate_box': CoordinateBoxNode,
    'com.cad.datum_plane': DatumPlaneNode,

    # 2D Sketching
    'com.cad.sketch': SketchNode,
    'com.cad.rect_sketch': RectangleSketchNode,
    'com.cad.circle_sketch': CircleSketchNode,
    'com.cad.sketch.line': LineSketchNode,
    'com.cad.sketch.arc': ArcSketchNode,
    'com.cad.sketch.circle': ParametricCircleSketchNode,
    'com.cad.sketch.rectangle': ParametricRectangleSketchNode,
    'com.cad.sketch.polygon': PolygonSketchNode,
    'com.cad.line': LineNode,
    'com.cad.arc': ArcNode,
    'com.cad.polygon': PolygonNode,
    'com.cad.spline': SplineNode,
    'com.cad.ellipse': EllipseNode,

    # 3D Operations
    'com.cad.extrude': ExtrudeNode,
    'com.cad.revolve': RevolveNode,
    'com.cad.sweep': SweepNode,
    'com.cad.loft': LoftNode,
    'com.cad.helix': HelixNode,

    # Cutting Operations
    'com.cad.pocket': PocketNode,
    'com.cad.cut_extrude': CutExtrudeNode,
    'com.cad.cylinder_cut': CylinderCutNode,
    'com.cad.hole_at_coords': HoleAtCoordinatesNode,
    'com.cad.multi_hole': MultiHoleNode,
    'com.cad.rectangular_cut': RectangularCutNode,
    'com.cad.slot_cut': SlotCutNode,
    'com.cad.array_holes': ArrayHolesNode,

    # Modifications
    'com.cad.fillet': FilletNode,
    'com.cad.chamfer': ChamferNode,
    'com.cad.shell': ShellNode,
    'com.cad.offset': OffsetNode,
    'com.cad.draft': DraftNode,

    # Boolean
    'com.cad.boolean': BooleanNode,

    # Transformations
    'com.cad.translate': TranslateNode,
    'com.cad.rotate': RotateNode,
    'com.cad.scale': ScaleNode,
    'com.cad.mirror': MirrorNode,

    # Patterns
    'com.cad.linear_pattern': LinearPatternNode,
    'com.cad.circular_pattern': CircularPatternNode,
    'com.cad.pattern.radial': RadialPatternNode,
    'com.cad.pattern.mirror': MirrorPatternNode,
    'com.cad.pattern.grid': GridPatternNode,

    # Selection
    'com.cad.select_face': SelectFaceNode,

    # Constraints
    'com.cad.constraint_distance': DistanceConstraintNode,
    'com.cad.constraint_angle': AngleConstraintNode,
    'com.cad.constraint_coincident': CoincidentConstraintNode,

    # Assembly
    'com.cad.assembly': AssemblyNode,
    'com.cad.mate': MateNode,

    # Analysis
    'com.cad.mass_properties': MassPropertiesNode,
    'com.cad.stress_analysis': StressAnalysisNode,
    'com.cad.bounding_box': BoundingBoxNode,
    'com.cad.volume': VolumeNode,
    'com.cad.surface_area': SurfaceAreaNode,
    'com.cad.center_of_mass': CenterOfMassNode,

    # Simulation
    'com.cad.force': ForceNode,
    'com.cad.fixed_support': FixedSupportNode,
    'com.cad.simulation_run': SimulationRunNode,

    # FEA Simulation (New)
    'com.cad.sim.material': MaterialNode,
    'com.cad.sim.mesh': MeshNode,
    'com.cad.sim.constraint': ConstraintNode,
    'com.cad.sim.load': LoadNode,
    'com.cad.sim.pressure_load': PressureLoadNode,
    'com.cad.sim.solver': SolverNode,
    'com.cad.sim.topopt': TopologyOptimizationNode,

    # Advanced Features
    'com.cad.text': TextNode,
    'com.cad.thread': ThreadNode,
    'com.cad.split': SplitNode,

    # Measurement
    'com.cad.measure_distance': MeasureDistanceNode,

    # Parameters / IO
    'com.cad.number': NumberNode,
    'com.cad.variable': VariableNode,

    # Export
    'com.cad.drawing': DrawingNode,
    'com.cad.property_table': PropertyTableNode,
    'com.cad.report': ReportGeneratorNode,
    'com.cad.export_step': ExportStepNode,
    'com.cad.export_stl': ExportStlNode,
}

# Mapping of Class Name -> Node Class (for legacy loading)
NODE_NAME_MAPPING = {cls.__name__: cls for cls in NODE_CLASS_MAPPING.values()}
