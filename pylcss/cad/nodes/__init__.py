# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
CAD Nodes - All node implementations consolidated.

Structure:
    nodes/
    ├── primitives.py    # Box, Cylinder, Sphere
    ├── operations.py    # Extrude, Revolve, Boolean, Fillet, etc.
    ├── cutting.py       # Hole, Pocket, Cut operations
    ├── patterns.py      # Linear, Circular, Grid patterns
    ├── sketch.py        # Sketch operations
    ├── simulation.py    # FEA, TopOpt, Material, Mesh
    ├── advanced.py      # Sketch, Assembly, Analysis
    ├── parametric.py    # Additional shapes, transforms
    ├── io.py            # Export STEP/STL
    └── values.py        # Number, Variable
"""

# Core base classes
from pylcss.cad.core.base_node import CadQueryNode, is_numeric, is_shape, resolve_numeric_input, resolve_shape_input
from pylcss.cad.core.registry import NODE_REGISTRY, register_node

# =============================================================================
# PRIMITIVES
# =============================================================================
from pylcss.cad.nodes.primitives import BoxNode, CylinderNode, SphereNode
from pylcss.cad.nodes.parametric import ConeNode, TorusNode, WedgeNode, PyramidNode

# =============================================================================
# SKETCHING
# =============================================================================
from pylcss.cad.nodes.advanced import SketchNode
from pylcss.cad.nodes.parametric import SplineNode, EllipseNode
from pylcss.cad.nodes.sketch import (
    LineSketchNode, ArcSketchNode, ParametricCircleSketchNode, 
    ParametricRectangleSketchNode, PolygonSketchNode
)

# =============================================================================
# 3D OPERATIONS
# =============================================================================
from pylcss.cad.nodes.operations import (
    ExtrudeNode, PocketNode, FilletNode, SelectFaceNode, 
    CutExtrudeNode, BooleanNode, RevolveNode, CylinderCutNode,
    ChamferNode, ShellNode
)
from pylcss.cad.nodes.parametric import SweepNode, LoftNode, HelixNode

# =============================================================================
# CUTTING OPERATIONS
# =============================================================================
from pylcss.cad.nodes.cutting import (
    HoleAtCoordinatesNode, MultiHoleNode, RectangularCutNode, 
    SlotCutNode, ArrayHolesNode
)

# =============================================================================
# MODIFICATIONS
# =============================================================================
from pylcss.cad.nodes.parametric import OffsetNode

# =============================================================================
# TRANSFORMS
# =============================================================================
from pylcss.cad.nodes.parametric import TranslateNode, RotateNode, ScaleNode, MirrorNode

# =============================================================================
# PATTERNS
# =============================================================================
from pylcss.cad.nodes.parametric import LinearPatternNode, CircularPatternNode
from pylcss.cad.nodes.patterns import RadialPatternNode, MirrorPatternNode, GridPatternNode

# =============================================================================
# ASSEMBLY
# =============================================================================
from pylcss.cad.nodes.advanced import AssemblyNode

# =============================================================================
# ANALYSIS
# =============================================================================
from pylcss.cad.nodes.advanced import MassPropertiesNode, BoundingBoxNode

# =============================================================================
# SIMULATION (FEA & TopOpt)
# =============================================================================
from pylcss.cad.nodes.simulation import (
    MaterialNode, MeshNode, ConstraintNode, LoadNode, PressureLoadNode,
    SolverNode, TopologyOptimizationNode
)

# =============================================================================
# IO
# =============================================================================
from pylcss.cad.nodes.io import ExportStepNode, ExportStlNode
from pylcss.cad.nodes.values import NumberNode, VariableNode

# =============================================================================
# ALL EXPORTS (58 nodes)
# =============================================================================
__all__ = [
    # Core
    "CadQueryNode", "NODE_REGISTRY", "register_node",
    "is_numeric", "is_shape", "resolve_numeric_input", "resolve_shape_input",
    
    # Primitives (7)
    "BoxNode", "CylinderNode", "SphereNode", "ConeNode", "TorusNode", 
    "WedgeNode", "PyramidNode",
    
    # Sketching (8)
    "SketchNode", "SplineNode", "EllipseNode",
    "LineSketchNode", "ArcSketchNode", "ParametricCircleSketchNode",
    "ParametricRectangleSketchNode", "PolygonSketchNode",
    
    # Operations (13)
    "ExtrudeNode", "PocketNode", "FilletNode", "SelectFaceNode",
    "CutExtrudeNode", "BooleanNode", "RevolveNode", "CylinderCutNode",
    "ChamferNode", "ShellNode", "SweepNode", "LoftNode", "HelixNode",
    
    # Cutting (5)
    "HoleAtCoordinatesNode", "MultiHoleNode", "RectangularCutNode",
    "SlotCutNode", "ArrayHolesNode",
    
    # Modifications (1)
    "OffsetNode",
    
    # Transforms (4)
    "TranslateNode", "RotateNode", "ScaleNode", "MirrorNode",
    
    # Patterns (5)
    "LinearPatternNode", "CircularPatternNode", 
    "RadialPatternNode", "MirrorPatternNode", "GridPatternNode",
    
    # Assembly (1)
    "AssemblyNode",
    
    # Analysis (2)
    "MassPropertiesNode", "BoundingBoxNode",
    
    # Simulation (7)
    "MaterialNode", "MeshNode", "ConstraintNode", "LoadNode", "PressureLoadNode",
    "SolverNode", "TopologyOptimizationNode",
    
    # IO (4)
    "ExportStepNode", "ExportStlNode", "NumberNode", "VariableNode",
]
