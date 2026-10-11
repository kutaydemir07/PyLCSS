# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import cadquery as cq
from pylcss.cae.design_studio.core.base_node import CadQueryNode

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
        self.add_output('shape', color=(100, 255, 100))
        self.create_property('assembly_name', 'Assembly1', widget_type='string')

    def run(self):
        self.clear_error()
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

        if not parts:
            self.set_error("Connect at least one CAD part to the assembly.")
            return None
        
        # Standard Assembly path: groups parts into an assembly structure
        asm = cq.Assembly(name=asm_name)
        for idx, part in enumerate(parts):
            name = f"part_{idx+1}"
            asm.add(part, name=name)
            
        return asm
