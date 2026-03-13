# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
CAD Nodes — all node implementations.

Structure:
    nodes/
    ├── geometry.py           # Primitives (Box, Cylinder, Sphere, …)
    ├── sketcher.py           # Sketching (Line, Arc, Circle, …)
    ├── modeling.py           # 3-D ops (Extrude, Revolve, Boolean, Fillet, Transform)
    ├── surfacing.py          # Surfacing (Sweep, Loft, Helix)
    ├── features.py           # Engineering features (Holes, Cuts)
    ├── analysis.py           # Mass props, Bounding box
    ├── assembly.py           # Assembly
    ├── patterns.py           # Patterns
    ├── io.py                 # IO
    ├── values.py             # Parameters
    ├── advanced.py           # Import/Export, Thicken, Split, Pipe, Text, Math, Measure
    ├── fem/                  # FEM simulation sub-package
    │   ├── _helpers.py       #   Shared helpers, filters, constants (lam_lame, MATERIAL_DATABASE)
    │   ├── materials.py      #   MaterialNode
    │   ├── mesh.py           #   MeshNode
    │   ├── boundary_conditions.py  # ConstraintNode, LoadNode, PressureLoadNode
    │   ├── solver.py         #   SolverNode (static linear FEA)
    │   ├── topology_optimization.py  # TopologyOptimizationNode (SIMP)
    │   └── advanced.py       #   RemeshNode, SizeOptimizationNode, ShapeOptimizationNode
    └── crash/                # Crash / impact simulation sub-package
        ├── _mechanics.py     #   Low-level solid-mechanics helpers (B-matrix, plasticity, contact)
        ├── _gpu_kernels.py   #   Taichi/GPU Neo-Hookean + J2 kernels
        ├── materials.py      #   CrashMaterialNode
        ├── conditions.py     #   ImpactConditionNode
        ├── solver.py         #   CrashSolverNode (CPU explicit transient)
        └── solver_gpu.py     #   CrashSolverGPUNode (Taichi GPU)
"""

# Core base classes
from pylcss.cad.core.base_node import CadQueryNode, is_numeric, is_shape, resolve_numeric_input, resolve_shape_input
from pylcss.cad.core.registry import NODE_REGISTRY, register_node

# =============================================================================
# GEOMETRY (Primitives)
# =============================================================================
from pylcss.cad.nodes.geometry import (
    BoxNode, CylinderNode, SphereNode, ConeNode, TorusNode, 
    WedgeNode, PyramidNode
)

# =============================================================================
# SKETCHER
# =============================================================================
from pylcss.cad.nodes.sketcher import (
    SketchNode, SplineNode, EllipseNode, PolylineNode,
    LineSketchNode, ArcSketchNode, ParametricCircleSketchNode, 
    ParametricRectangleSketchNode, PolygonSketchNode
)

# =============================================================================
# MODELING (3D Ops & Transforms)
# =============================================================================
from pylcss.cad.nodes.modeling import (
    ExtrudeNode, RevolveNode, BooleanNode, FilletNode, 
    ChamferNode, ShellNode, SelectFaceNode, InteractiveSelectFaceNode, OffsetNode,
    TranslateNode, RotateNode, ScaleNode, MirrorNode,
    CutExtrudeNode, TwistedExtrudeNode
)

# =============================================================================
# SURFACING
# =============================================================================
from pylcss.cad.nodes.surfacing import (
    SweepNode, LoftNode, HelixNode
)

# =============================================================================
# FEATURES (Cuts & Holes)
# =============================================================================
from pylcss.cad.nodes.features import (
    PocketNode, CylinderCutNode,
    HoleAtCoordinatesNode, MultiHoleNode, RectangularCutNode, 
    SlotCutNode, ArrayHolesNode
)

# =============================================================================
# PATTERNS (Advanced)
# =============================================================================
from pylcss.cad.nodes.patterns import (
    RadialPatternNode, MirrorPatternNode, GridPatternNode,
    LinearPatternNode, CircularPatternNode
)

# =============================================================================
# ASSEMBLY
# =============================================================================
from pylcss.cad.nodes.assembly import AssemblyNode

# =============================================================================
# ANALYSIS
# =============================================================================
from pylcss.cad.nodes.analysis import MassPropertiesNode, BoundingBoxNode

# =============================================================================
# FEM / SIMULATION
# =============================================================================
from pylcss.cad.nodes.fem import (
    MaterialNode, MeshNode, ConstraintNode, LoadNode, PressureLoadNode,
    SolverNode, TopologyOptimizationNode
)

# =============================================================================
# IO & VALUES
# =============================================================================
from pylcss.cad.nodes.io import ExportStepNode, ExportStlNode
from pylcss.cad.nodes.values import NumberNode, VariableNode

# =============================================================================
# ADVANCED NODES
# =============================================================================
from pylcss.cad.nodes.advanced import (
    ImportStepNode, ImportStlNode, ThickenNode, SplitNode,
    PipeNode, TextNode, MathExpressionNode,
    MeasureDistanceNode, SurfaceAreaNode,
)

# =============================================================================
# ADVANCED FEA (previously unregistered)
# =============================================================================
from pylcss.cad.nodes.fem import (
    RemeshNode, SizeOptimizationNode, ShapeOptimizationNode,
)

# =============================================================================
# CRASH / IMPACT SIMULATION
# =============================================================================
from pylcss.cad.nodes.crash import (
    CrashMaterialNode,
    ImpactConditionNode,
    CrashSolverNode,
    CrashSolverGPUNode,
)

# =============================================================================
# ALL EXPORTS
# =============================================================================
__all__ = [
    # Core
    "CadQueryNode", "NODE_REGISTRY", "register_node",
    "is_numeric", "is_shape", "resolve_numeric_input", "resolve_shape_input",
    
    # Geometry
    "BoxNode", "CylinderNode", "SphereNode", "ConeNode", "TorusNode", 
    "WedgeNode", "PyramidNode",
    
    # Sketcher
    "SketchNode", "SplineNode", "EllipseNode", "PolylineNode",
    "LineSketchNode", "ArcSketchNode", "ParametricCircleSketchNode",
    "ParametricRectangleSketchNode", "PolygonSketchNode",
    
    # Modeling
    "ExtrudeNode", "RevolveNode", "BooleanNode", "FilletNode", 
    "ChamferNode", "ShellNode", "SelectFaceNode", "InteractiveSelectFaceNode", "OffsetNode",
    "TranslateNode", "RotateNode", "ScaleNode", "MirrorNode",
    "LinearPatternNode", "CircularPatternNode", "TwistedExtrudeNode",
    
    # Surfacing
    "SweepNode", "LoftNode", "HelixNode",
    
    # Features
    "PocketNode", "CutExtrudeNode", "CylinderCutNode",
    "HoleAtCoordinatesNode", "MultiHoleNode", "RectangularCutNode",
    "SlotCutNode", "ArrayHolesNode",
    
    # Patterns
    "RadialPatternNode", "MirrorPatternNode", "GridPatternNode",
    
    # Assembly
    "AssemblyNode",
    
    # Analysis
    "MassPropertiesNode", "BoundingBoxNode",
    
    # FEM
    "MaterialNode", "MeshNode", "ConstraintNode", "LoadNode", "PressureLoadNode",
    "SolverNode", "TopologyOptimizationNode",
    
    # IO / Values
    "ExportStepNode", "ExportStlNode", "NumberNode", "VariableNode",
    
    # Advanced
    "ImportStepNode", "ImportStlNode", "ThickenNode", "SplitNode",
    "PipeNode", "TextNode", "MathExpressionNode",
    "MeasureDistanceNode", "SurfaceAreaNode",
    
    # Advanced FEA
    "RemeshNode", "SizeOptimizationNode", "ShapeOptimizationNode",

    # Crash / Impact
    "CrashMaterialNode", "ImpactConditionNode", "CrashSolverNode",
    "CrashSolverGPUNode",
]
