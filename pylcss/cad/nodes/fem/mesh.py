# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM mesh node — tetrahedral mesh generation via Netgen-OCC."""
import os
import tempfile
import logging
from skfem import *
from pylcss.cad.core.base_node import CadQueryNode
from pylcss.cad.nodes.fem._helpers import suppress_output, OCCGeometry

logger = logging.getLogger(__name__)

class MeshNode(CadQueryNode):
    """Generates a finite element mesh from a shape using Netgen."""
    __identifier__ = 'com.cad.sim.mesh'
    NODE_NAME = 'Generate Mesh (Netgen)'

    def __init__(self):
        super().__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('element_size', color=(180, 180, 0))
        # NEW: Local refinement inputs
        self.add_input('refinement_faces', color=(255, 100, 100))  # List of faces for refinement
        self.add_input('refinement_size', color=(255, 100, 100))   # Smaller element size for refinement
        self.add_output('mesh', color=(200, 100, 200))
        
        # Mesh type selection
        self.create_property('mesh_type', 'Tet', widget_type='combo',
                             items=['Tet', 'Tet10'])
        self.create_property('element_size', 2.0, widget_type='float')
        self.create_property('refinement_size', 0.5, widget_type='float')  # Finer mesh for critical areas

    def run(self):
        if OCCGeometry is None:
            self.set_error("Netgen-occ is not installed")
            return None

        shape = self.get_input_shape('shape')
        # Resolve element size input with fallback to property
        size = self.get_input_value('element_size', 'element_size')
        size = float(size)
        
        # NEW: Get refinement parameters
        refinement_faces = self.get_input_value('refinement_faces', None)
        refinement_size = self.get_input_value('refinement_size', 'refinement_size')
        refinement_size = float(refinement_size)
        
        if not shape:
            return None

        # Handle assemblies by converting to compound
        if hasattr(shape, 'toCompound'):
            try:
                shape = shape.toCompound()
            except Exception:
                return None

        # Optimized temporary file handling for performance
        # Try to use RAM disk if available (significant speedup for optimization loops)
        temp_base = None
        try:
            # Check for common RAM disk locations
            ram_disk_paths = ['R:\\', 'Z:\\', '/tmp/', '/dev/shm/']
            for path in ram_disk_paths:
                if os.path.exists(path) and os.access(path, os.W_OK):
                    temp_base = path
                    break
            
            # Fallback to system temp directory
            if temp_base is None:
                temp_base = tempfile.gettempdir()
            
            # Initialise paths before try so the finally block can safely
            # reference them even if the NamedTemporaryFile call fails.
            step_path = None
            msh_path  = None

            # Create temporary files in optimized location
            with tempfile.NamedTemporaryFile(suffix=".step", dir=temp_base, delete=False) as step_file:
                step_path = step_file.name

            msh_path = step_path.replace(".step", ".msh")

            try:
                # 1. Export CadQuery shape to STEP
                if hasattr(shape, 'val'):
                    shape.val().exportStep(step_path)
                else:
                    shape.exportStep(step_path)
                
                # 2. Load Geometry with Netgen and generate mesh (suppress verbose output)
                with suppress_output():
                    geo = OCCGeometry(step_path)
                    
                    # NEW: Apply local mesh refinement if specified
                    if refinement_faces is not None:
                        try:
                            # Handle SelectFaceNode dict format: {'workplane': ..., 'face': ..., 'faces': [...]}
                            if isinstance(refinement_faces, dict):
                                face_list = refinement_faces.get('faces', [])
                                if not face_list and refinement_faces.get('face') is not None:
                                    face_list = [refinement_faces['face']]
                            elif hasattr(refinement_faces, 'vals'):
                                face_list = refinement_faces.vals()
                            else:
                                face_list = [refinement_faces]
                            
                            for face in face_list:
                                if hasattr(face, 'hashCode'):
                                    # Set finer mesh size on specific faces
                                    geo.SetFaceMaxH(face.hashCode(), refinement_size)
                        except Exception:
                            pass
                    
                    # 3. Generate Mesh
                    # maxh controls the global element size
                    ng_mesh = geo.GenerateMesh(maxh=size)
                    
                    # 4. Export to Gmsh format (Version 2 is most compatible with skfem/meshio)
                    # Netgen's Export function takes the filename and the format string
                    ng_mesh.Export(msh_path, "Gmsh2 Format")
                
                # 5. Load into skfem
                print("FEA Mesh: Loading into skfem...")
                mesh = Mesh.load(msh_path)
                print(f"FEA Mesh: Load complete. Nodes: {mesh.p.shape[1]}, Tets: {mesh.t.shape[1]}")
                
            except Exception as e:
                print(f"FEA Mesh: ERROR loading mesh: {e}")
                return None
                
            finally:
                # Clean up temporary files immediately.
                # Guard against step_path / msh_path being None when the
                # NamedTemporaryFile call itself failed (UnboundLocalError fix).
                try:
                    if step_path and os.path.exists(step_path):
                        os.remove(step_path)
                    if msh_path and os.path.exists(msh_path):
                        os.remove(msh_path)
                except OSError:
                    pass  # Ignore cleanup errors
        
        except Exception:
            return None
        
        return mesh

