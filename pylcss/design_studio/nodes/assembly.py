# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import cadquery as cq
from pylcss.design_studio.core.base_node import CadQueryNode

class AssemblyNode(CadQueryNode):
    """Group CAD bodies without defining simulation contact."""
    __identifier__ = 'com.cad.assembly.assembly'
    NODE_NAME = 'Group Bodies'

    def __init__(self):
        super(AssemblyNode, self).__init__()
        self.add_input(
            'bodies',
            color=(100, 255, 100),
            multi_input=True,
        )
        # Fixed slots are retained, hidden, and still executed so old projects
        # load unchanged. New graphs use the unlimited Bodies input above.
        for index in range(1, 5):
            legacy_input = self.add_input(
                f'part_{index}',
                display_name=False,
                color=(100, 255, 100),
            )
            legacy_input.view.setVisible(False)
        
        self.add_output('shape', color=(100, 255, 100))
        legacy_output = self.add_output(
            'assembly', display_name=False, color=(200, 150, 100)
        )
        legacy_output.view.setVisible(False)
        self.create_property('assembly_name', 'Assembly1', widget_type='string')
        # Compatibility only. Boolean is the public tool for fusing geometry.
        self.create_property('fuse_parts', False, widget_type='bool')

    def run(self):
        self.clear_error()
        from pylcss.input_values import as_bool

        fuse = as_bool(self.get_property('fuse_parts'))
        asm_name = str(self.get_property('assembly_name') or '').strip() or 'Assembly'
        
        parts = []
        connected_bodies = self.get_input_list('bodies')
        for val in connected_bodies:
            if val is None:
                continue
            if hasattr(val, 'val'):
                parts.append(val.val())
            elif isinstance(val, cq.Assembly):
                parts.append(val.toCompound())
            else:
                parts.append(val)
        for i in range(1, 5):
            val = self.get_input_value(f'part_{i}', None)
            if val is not None:
                # Extract raw shape if it's a Workplane or wrapper
                if hasattr(val, 'val'):
                    parts.append(val.val())
                elif isinstance(val, cq.Assembly):
                    # For nested assemblies, toCompound() gets everything
                    parts.append(val.toCompound())
                else:
                    parts.append(val)

        if not parts:
            self.set_error("Connect at least one CAD part to the assembly.")
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
                return None
        
        # Standard Assembly path
        asm = cq.Assembly(name=asm_name)
        for idx, part in enumerate(parts):
            name = f"part_{idx+1}"
            asm.add(part, name=name)
            
        return asm
