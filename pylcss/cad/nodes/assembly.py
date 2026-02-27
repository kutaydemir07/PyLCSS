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
        self.create_property('fuse_parts', False, widget_type='bool')

    def run(self):
        fuse = self.get_property('fuse_parts')
        asm_name = self.get_property('assembly_name')
        
        parts = []
        for i in range(1, 5):
            val = self.get_input_value(f'part_{i}', None)
            if val:
                # Extract raw shape if it's a Workplane or wrapper
                if hasattr(val, 'val'):
                    parts.append(val.val())
                elif isinstance(val, cq.Assembly):
                    # For nested assemblies, toCompound() gets everything
                    parts.append(val.toCompound())
                else:
                    parts.append(val)

        if not parts:
            return None
            
        if fuse:
            # Union all parts into a single compound
            try:
                fused = parts[0]
                for next_part in parts[1:]:
                    fused = fused.union(next_part)
                
                # Wrap in a fresh assembly so downstream nodes (Mesh, SelectFace) 
                # still receive the expected type.
                final_asm = cq.Assembly(name=asm_name)
                final_asm.add(fused, name="Fused_Body")
                return final_asm
            except Exception as e:
                self.set_error(f"Fusion failed: {e}")
                # Fallback to standard assembly
        
        # Standard Assembly path
        asm = cq.Assembly(name=asm_name)
        for idx, part in enumerate(parts):
            name = f"part_{idx+1}"
            asm.add(part, name=name)
            
        return asm
