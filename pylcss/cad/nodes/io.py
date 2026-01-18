# Copyright (c) 2026 Kutay Demir.
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
        
        if shape:
            fname = self.get_property('filename')
            if not fname.endswith(".step"):
                fname += ".step"
            
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
                return True
            except Exception:
                return False
        return False


class ExportStlNode(CadQueryNode):
    """
    Exports the result to an STL file (binary format).
    
    Supports:
    - CadQuery shapes (Workplane, Solid, Compound)  
    - TopOpt mesh with thresholded surface extraction (exact GUI match)
    - Optional mesh smoothing for organic, manufacturable shapes
    
    Note: All dimensions are in millimeters (mm). If your CAD geometry
    uses different units, the exported STL will reflect those units directly.
    """
    __identifier__ = 'com.cad.export_stl'
    NODE_NAME = 'Export STL'

    def __init__(self):
        super(ExportStlNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.create_property('filename', 'output.stl', widget_type='string')
        # Smoothing options for TopOpt mesh
        self.create_property('smoothing', 10, widget_type='int')  # Number of smoothing iterations (0=off)

    def run(self):
        import numpy as np
        from pylcss.cad.core.base_node import resolve_any_input
        
        # Get input - try both shape resolution and generic input
        port = self.get_input('shape')
        shape = None
        
        # First try resolve_any_input to handle dict outputs from TopOpt
        if port and port.connected_ports():
            shape = resolve_any_input(port)
        
        # Fallback to standard shape resolution
        if shape is None:
            shape = self.get_input_shape('shape')
        
        
        if shape is None:
            return False
            
        fname = self.get_property('filename')
        if not fname.endswith(".stl"):
            fname += ".stl"
        
        try:
            vertices = None
            faces = None
            
            # Case 1: TopOpt result dict - extract thresholded surface (exact GUI match)
            if isinstance(shape, dict) and 'mesh' in shape and 'density' in shape:
                mesh = shape['mesh']
                density = np.asarray(shape['density'])
                cutoff = float(shape.get('density_cutoff', 0.3))
                
                
                vertices, faces = self._extract_thresholded_surface(mesh, density, cutoff)
            
            # Case 2: Direct mesh dict with vertices/faces (e.g., from recovered_shape output)
            elif isinstance(shape, dict):
                if 'vertices' in shape and 'faces' in shape:
                    vertices = np.array(shape['vertices'])
                    faces = np.array(shape['faces'])
                # Case 2b: Nested recovered_shape
                elif 'recovered_shape' in shape and shape['recovered_shape'] is not None:
                    rs = shape['recovered_shape']
                    if 'vertices' in rs and 'faces' in rs:
                        vertices = np.array(rs['vertices'])
                        faces = np.array(rs['faces'])
            
            # Case 3: CadQuery shape - tessellate it
            if vertices is None and hasattr(shape, 'tessellate'):
                topo_shape = shape
                if hasattr(topo_shape, 'val'):
                    try:
                        topo_shape = topo_shape.val()
                    except:
                        pass
                
                triangulation = topo_shape.tessellate(tolerance=0.01, angularTolerance=0.1)
                if isinstance(triangulation, dict):
                    verts_list = triangulation.get('vertices') or triangulation.get('verts')
                    tris_list = triangulation.get('triangles') or triangulation.get('faces')
                else:
                    verts_list, tris_list = triangulation[0], triangulation[1]
                
                # Convert CadQuery Vector objects to numpy array
                vertices = np.array([[v.x, v.y, v.z] for v in verts_list])
                faces = np.array(tris_list)
            
            # Case 4: Use CadQuery exporter as fallback
            if vertices is None:
                if hasattr(shape, 'val'):
                    shape_to_export = shape.val()
                else:
                    shape_to_export = shape
                cq.exporters.export(shape_to_export, fname)
                return True
            
            # Apply mesh smoothing for organic shapes (if enabled and we have mesh data)
            smoothing_iters = int(self.get_property('smoothing'))
            if smoothing_iters > 0 and vertices is not None and len(faces) > 0:
                vertices = self._taubin_smooth(vertices, faces, smoothing_iters)
            
            # Write binary STL using raw NumPy (no numpy-stl dependency)
            self._write_binary_stl(fname, vertices, faces)
            return True
            
        except Exception:
            return False
    
    def _extract_thresholded_surface(self, mesh, density, cutoff):
        """
        Extract surface triangles from thresholded tetrahedra.
        This exactly matches what the VTK viewer shows.
        
        Algorithm:
        1. Keep only tetrahedra where density >= cutoff
        2. For each tet, extract its 4 triangular faces
        3. Keep only faces that appear exactly once (boundary faces)
        """
        import numpy as np
        from collections import Counter
        
        # mesh.p is (3, N_vertices), mesh.t is (4, N_tets)
        pts = mesh.p  # (3, N)
        tets = mesh.t  # (4, M)
        
        # Filter tetrahedra by density threshold
        mask = density >= cutoff
        kept_tets = tets[:, mask]  # (4, K) where K = number of kept tets
        n_kept = kept_tets.shape[1]
        
        if n_kept == 0:
            return None, None
        
        
        # Extract all faces from kept tetrahedra
        # Each tet has 4 triangular faces: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        face_indices = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3)
        ]
        
        all_faces = []
        for i in range(n_kept):
            tet = kept_tets[:, i]
            for fi in face_indices:
                # Sort vertex indices to create a canonical face key
                face = tuple(sorted([tet[fi[0]], tet[fi[1]], tet[fi[2]]]))
                all_faces.append(face)
        
        # Count face occurrences - boundary faces appear exactly once
        face_counts = Counter(all_faces)
        boundary_faces = [face for face, count in face_counts.items() if count == 1]
        
        
        if len(boundary_faces) == 0:
            return None, None
        
        # Get unique vertex indices used in boundary faces
        used_verts = set()
        for face in boundary_faces:
            used_verts.update(face)
        used_verts = sorted(used_verts)
        
        # Create vertex mapping: old_index -> new_index
        vert_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_verts)}
        
        # Extract vertices
        vertices = pts[:, used_verts].T  # (N_used, 3)
        
        # Remap face indices
        faces = np.array([[vert_map[v] for v in face] for face in boundary_faces])
        
        return vertices, faces
    
    def _taubin_smooth(self, vertices, faces, iterations=10, lambda_factor=0.5, mu_factor=-0.53):
        """
        Taubin mesh smoothing - volume-preserving Laplacian smoothing.
        
        Unlike simple Laplacian smoothing which shrinks the mesh, Taubin alternates
        between positive (smoothing) and negative (inflation) steps to maintain volume.
        
        Parameters:
        -----------
        vertices : ndarray (N, 3) - vertex positions
        faces : ndarray (M, 3) - triangle indices
        iterations : int - number of smoothing iterations
        lambda_factor : float - smoothing factor (default 0.5)
        mu_factor : float - inflation factor (default -0.53, calculated from lambda)
        
        Returns:
        --------
        smoothed_vertices : ndarray (N, 3)
        """
        import numpy as np
        from collections import defaultdict
        
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces, dtype=np.int32)
        n_verts = len(vertices)
        
        # Build adjacency: for each vertex, find its neighbors
        adjacency = defaultdict(set)
        for face in faces:
            for i in range(3):
                v1 = face[i]
                v2 = face[(i + 1) % 3]
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
        
        # Convert to lists for faster iteration
        neighbors = [list(adjacency[i]) for i in range(n_verts)]
        
        # Taubin smoothing iterations
        for _ in range(iterations):
            # Step 1: Laplacian smoothing with positive lambda
            new_verts = vertices.copy()
            for i in range(n_verts):
                if len(neighbors[i]) > 0:
                    neighbor_avg = np.mean(vertices[neighbors[i]], axis=0)
                    new_verts[i] = vertices[i] + lambda_factor * (neighbor_avg - vertices[i])
            vertices = new_verts
            
            # Step 2: Laplacian "unsmoothing" with negative mu (prevents shrinkage)
            new_verts = vertices.copy()
            for i in range(n_verts):
                if len(neighbors[i]) > 0:
                    neighbor_avg = np.mean(vertices[neighbors[i]], axis=0)
                    new_verts[i] = vertices[i] + mu_factor * (neighbor_avg - vertices[i])
            vertices = new_verts
        
        return vertices
    
    def _write_binary_stl(self, filename, vertices, faces):
        """
        Write a binary STL file using raw NumPy.
        
        Binary STL format:
        - 80 bytes: header
        - 4 bytes: uint32 triangle count
        - For each triangle (50 bytes each):
            - 12 bytes: normal vector (3 x float32)
            - 36 bytes: 3 vertices (9 x float32)
            - 2 bytes: attribute byte count (uint16, usually 0)
        """
        import numpy as np
        import struct
        
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        n_triangles = len(faces)
        
        with open(filename, 'wb') as f:
            # Header (80 bytes)
            header = b'Binary STL exported by PyLCSS (units: mm)' + b'\0' * 40
            f.write(header[:80])
            
            # Triangle count
            f.write(struct.pack('<I', n_triangles))
            
            # Triangles
            for face in faces:
                # Get vertices for this face
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                
                # Calculate normal (cross product)
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-10:
                    normal = normal / norm_len
                else:
                    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                
                # Write normal (3 floats)
                f.write(struct.pack('<3f', *normal))
                
                # Write vertices (9 floats)
                f.write(struct.pack('<3f', *v0))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))
                
                # Attribute byte count (uint16)
                f.write(struct.pack('<H', 0))


