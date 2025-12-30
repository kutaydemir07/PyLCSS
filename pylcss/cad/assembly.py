# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Simple assembly helper using CadQuery's Assembly.
"""
from typing import Any, Tuple, Optional

try:
    import cadquery as cq
    from cadquery import Assembly, Color
except ImportError:
    cq = None
    Assembly = None
    Color = None

class SimpleAssembly:
    """Wrapper for managing CadQuery Assemblies."""
    
    def __init__(self):
        if Assembly is None:
            raise RuntimeError("CadQuery Assembly is not available. Install cadquery.")
        self._asm = Assembly()

    def add(self, obj: Any, name: str, 
            loc: Tuple[float, float, float] = (0, 0, 0),
            rotate: Tuple[float, float, float] = (0, 0, 0),
            color: Optional[Tuple[float, float, float, float]] = None) -> None:
        """
        Add a Workplane/solid to the assembly.
        
        Args:
            obj: The CadQuery object (Workplane or Shape).
            name: Unique name for this part.
            loc: Translation (x, y, z).
            rotate: Rotation in degrees (rx, ry, rz).
            color: Optional color tuple (r, g, b, alpha) values 0.0-1.0.
        """
        # 1. Convert Workplane to Solid/Shape if needed
        shape = obj
        if hasattr(obj, "val"):
            shape = obj.val()

        # 2. Create Location (Translation + Rotation)
        # Your previous code ignored the 'rotate' variable!
        try:
            # Create translation vector
            vec_loc = cq.Vector(*loc)
            
            # Create rotation vector (assuming degrees)
            # cq.Location can take (translation, rotation_axis, angle) 
            # OR simple creation. A robust way is composing locations:
            
            # Location = Translate * RotateX * RotateY * RotateZ
            loc_obj = cq.Location(vec_loc)
            
            # Apply rotations (order matters, typically XYZ or ZYX)
            if rotate != (0, 0, 0):
                rx, ry, rz = rotate
                # Compose rotations
                loc_obj = loc_obj * cq.Location(cq.Vector(0,0,0), cq.Vector(1,0,0), rx)
                loc_obj = loc_obj * cq.Location(cq.Vector(0,0,0), cq.Vector(0,1,0), ry)
                loc_obj = loc_obj * cq.Location(cq.Vector(0,0,0), cq.Vector(0,0,1), rz)

        except Exception as e:
            print(f"Assembly Location Error: {e}")
            # Fallback to simple tuple if complex location fails
            loc_obj = cq.Location(cq.Vector(*loc))

        # 3. Handle Color
        cq_color = None
        if color and Color:
            # Color expects (r, g, b, a)
            cq_color = Color(*color)

        # 4. Add to Assembly
        self._asm.add(shape, name=name, loc=loc_obj, color=cq_color)

    def to_compound(self):
        """Return a compound suitable for exporting."""
        try:
            return self._asm.toCompound()
        except Exception:
            return self._asm

    @property
    def assembly(self):
        return self._asm