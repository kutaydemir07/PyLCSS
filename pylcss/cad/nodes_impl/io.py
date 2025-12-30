# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode

class ExportStepNode(CadQueryNode):
    """Exports the result to a STEP file."""
    __identifier__ = 'com.cad.export_step'
    NODE_NAME = 'Export STEP'

    def __init__(self):
        super(ExportStepNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.create_property('filename', 'output.step', widget_type='string')

    def run(self):
        shape = self.get_input_shape('shape')
        print(f"ExportStepNode: Input shape is {type(shape)}")
        
        if shape:
            fname = self.get_property('filename')
            if not fname.endswith(".step"):
                fname += ".step"
            
            print(f"ExportStepNode: Attempting to save to {fname}")
            try:
                # Convert Workplane to solid if needed
                if hasattr(shape, 'val'):
                    shape_to_export = shape.val()
                else:
                    shape_to_export = shape

                # Use CadQuery exporters which handle Compound/Shape/Workplane
                try:
                    cq.exporters.export(shape_to_export, fname)
                except Exception:
                    # Fallback to save() if available
                    if hasattr(shape_to_export, 'save'):
                        shape_to_export.save(fname)
                    else:
                        raise
                print(f"✓ SUCCESS: Saved {fname}")
                return True
            except Exception as e:
                print(f"✗ Export Error: {e}")
                return False
        else:
            print("ExportStepNode: No shape connected or shape is None")
        return False


class ExportStlNode(CadQueryNode):
    """Exports the result to an STL file."""
    __identifier__ = 'com.cad.export_stl'
    NODE_NAME = 'Export STL'

    def __init__(self):
        super(ExportStlNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.create_property('filename', 'output.stl', widget_type='string')

    def run(self):
        shape = self.get_input_shape('shape')
        
        if shape:
            fname = self.get_property('filename')
            if not fname.endswith(".stl"):
                fname += ".stl"
            
            try:
                # Convert Workplane to solid if needed
                if hasattr(shape, 'val'):
                    shape_to_export = shape.val()
                else:
                    shape_to_export = shape

                # Use CadQuery exporters for STL
                try:
                    cq.exporters.export(shape_to_export, fname)
                except Exception:
                    if hasattr(shape_to_export, 'save'):
                        shape_to_export.save(fname)
                    else:
                        raise
                print(f"✓ SUCCESS: Saved {fname}")
                return True
            except Exception as e:
                print(f"✗ Export Error: {e}")
                return False
        return False
