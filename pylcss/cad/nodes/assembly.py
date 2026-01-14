# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode

class AssemblyNode(CadQueryNode):
    """Create an assembly from multiple parts."""
    __identifier__ = 'com.cad.assembly.assembly'
    NODE_NAME = 'Assembly'

    def __init__(self):
        super(AssemblyNode, self).__init__()
        self.add_input('part_1', color=(100, 255, 100))
        self.add_input('part_2', color=(100, 255, 100))
        self.add_input('part_3', color=(100, 255, 100))
        self.add_input('part_4', color=(100, 255, 100))
        
        self.add_output('assembly', color=(200, 150, 100))
        self.create_property('assembly_name', 'Assembly1', widget_type='string')

    def run(self):
        asm = cq.Assembly(name=self.get_property('assembly_name'))
        
        parts = []
        # Dynamic inputs would be better, but fixed for now
        for i in range(1, 5):
            val = self.get_input_value(f'part_{i}', None)
            if val:
                # Handle different input types (Workplane, Shape, Assembly)
                if isinstance(val, cq.Workplane):
                    parts.append(val)
                elif hasattr(val, 'val'): # CadQuery object wrappers
                    parts.append(val)
                elif isinstance(val, cq.Assembly):
                    # Assemblies can be nested
                    parts.append(val)
                else:
                    # Try to add generic object if compatible
                    parts.append(val)

        if not parts:
            return None
            
        for idx, part in enumerate(parts):
            name = f"part_{idx+1}"
            # Extract name if part is Assembly
            if isinstance(part, cq.Assembly) and part.name:
                name = part.name
                
            asm.add(part, name=name)
            
        return asm
