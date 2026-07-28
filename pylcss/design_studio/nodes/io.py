# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import cadquery as cq
import os
import tempfile

from pylcss.design_studio.core.base_node import CadQueryNode


def _output_path(node, filename):
    """Resolve relative export paths beside the owning project when saved."""
    path = os.fspath(filename)
    if not os.path.isabs(path) and getattr(node, "_project_dir", None):
        path = os.path.join(node._project_dir, path)
    return os.path.abspath(path)

class ExportStepNode(CadQueryNode):
    """Exports the result to a STEP file."""
    __identifier__ = 'com.cad.export_step'
    NODE_NAME = 'Export STEP'

    def __init__(self):
        super(ExportStepNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('file', color=(180, 220, 255))
        self.create_property('filename', 'output.step', widget_type='string')

    def run(self):
        self.clear_error()
        shape = self.get_input_shape('shape')
        if shape is None:
            self.set_error("Connect a CAD shape to Export STEP.")
            return None

        fname = str(self.get_property('filename') or '').strip()
        if not fname:
            self.set_error("Choose a STEP output filename.")
            return None
        if not fname.lower().endswith((".step", ".stp")):
            fname += ".step"
        fname = _output_path(self, fname)
        os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)

        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".pylcss_step_",
                suffix=os.path.splitext(fname)[1],
                dir=os.path.dirname(fname) or ".",
            )
            os.close(fd)
            shape_to_export = shape.val() if hasattr(shape, 'val') else shape
            try:
                cq.exporters.export(shape_to_export, temp_path)
            except Exception:
                if hasattr(shape_to_export, 'save'):
                    shape_to_export.save(temp_path)
                else:
                    raise
            if not os.path.isfile(temp_path) or os.path.getsize(temp_path) == 0:
                raise RuntimeError("The STEP exporter did not create a non-empty file.")
            os.replace(temp_path, fname)
            temp_path = None
            return {'ok': True, 'file': fname, 'path': fname}
        except Exception as exc:
            self.set_error(f"STEP export failed: {exc}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


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
        self.add_output('file', color=(180, 220, 255))
        self.create_property('filename', 'output.stl', widget_type='string')
        # Smoothing options for TopOpt mesh
        self.create_property('smoothing', 0, widget_type='int')  # Number of smoothing iterations (0=off)

    def run(self):
        import numpy as np
        from pylcss.design_studio.core.base_node import resolve_any_input
        
        self.clear_error()
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
            self.set_error("Connect a CAD shape, recovered surface, or topology result to Export STL.")
            return None
            
        fname = str(self.get_property('filename') or '').strip()
        if not fname:
            self.set_error("Choose an STL output filename.")
            return None
        if not fname.lower().endswith(".stl"):
            fname += ".stl"
        fname = _output_path(self, fname)
        out_dir = os.path.dirname(fname)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        try:
            vertices = None
            faces = None
            
            # Case 1: TopOpt result dict - extract thresholded surface (exact GUI match)
            if isinstance(shape, dict) and 'mesh' in shape and 'density' in shape:
                mesh = shape['mesh']
                density = np.asarray(shape['density'], dtype=float).reshape(-1)
                cutoff = float(shape.get('density_cutoff', 0.3))
                if not np.all(np.isfinite(density)):
                    raise ValueError("Topology density contains NaN or infinite values.")
                if not np.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0:
                    raise ValueError("Density cutoff must be finite and between 0 and 1.")
                
                
                vertices, faces = self._extract_thresholded_surface(mesh, density, cutoff)
                if vertices is None or faces is None:
                    raise ValueError(
                        "No topology elements meet the current density cutoff; "
                        "lower the cutoff or rerun the optimization."
                    )
            
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
                    except Exception:
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
                if not os.path.isfile(fname) or os.path.getsize(fname) == 0:
                    raise RuntimeError("The STL exporter did not create a non-empty file.")
                return {'ok': True, 'file': fname, 'path': fname}
            
            # Apply mesh smoothing for organic shapes (if enabled and we have mesh data)
            smoothing_iters = int(self.get_property('smoothing'))
            if smoothing_iters < 0:
                raise ValueError("Smoothing iterations cannot be negative.")
            if smoothing_iters > 0 and vertices is not None and len(faces) > 0:
                vertices = self._taubin_smooth(vertices, faces, smoothing_iters)
            
            # Write binary STL using raw NumPy (no numpy-stl dependency).
            # The writer commits atomically so a failed export cannot destroy
            # an existing good result.
            self._write_binary_stl(fname, vertices, faces)
            return {
                'ok': True,
                'file': fname,
                'path': fname,
                'triangles': int(len(faces)) if faces is not None else 0,
                'vertices': int(len(vertices)) if vertices is not None else 0,
            }
            
        except Exception as exc:
            self.set_error(f"STL export failed: {exc}")
            return None
    
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
        from types import SimpleNamespace
        from pylcss.design_studio.nodes.selection import _mesh_boundary_face_data
        
        # mesh.p is (3, N_vertices), mesh.t is (4, N_tets)
        pts = mesh.p  # (3, N)
        tets = mesh.t  # (4, M)
        if np.asarray(tets).ndim != 2 or np.asarray(tets).shape[0] < 4:
            raise ValueError("Topology export requires tetrahedral mesh connectivity.")
        if density.size != np.asarray(tets).shape[1]:
            raise ValueError("Topology density count does not match the mesh elements.")
        
        # Filter tetrahedra by density threshold
        mask = density >= cutoff
        kept_tets = tets[:, mask]  # (4, K) where K = number of kept tets
        n_kept = kept_tets.shape[1]
        
        if n_kept == 0:
            return None, None
        
        
        data = _mesh_boundary_face_data(SimpleNamespace(p=pts, t=kept_tets))
        if data is None or len(data.get('faces', [])) == 0:
            return None, None
        boundary_faces = np.asarray(data['faces'], dtype=int)
        
        # Get unique vertex indices used in boundary faces
        used_verts = np.unique(boundary_faces.reshape(-1)).astype(int)
        
        # Create vertex mapping: old_index -> new_index
        vert_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(used_verts)}
        
        # Extract vertices
        vertices = pts[:, used_verts].T  # (N_used, 3)
        
        # Remap face indices
        faces = np.array([[vert_map[int(v)] for v in face] for face in boundary_faces])
        
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
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise ValueError("STL vertices must be an N x 3 array.")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
            raise ValueError("STL faces must be an M x 3 triangle array.")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("STL vertices contain NaN or infinite coordinates.")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("STL triangle connectivity contains invalid vertex indices.")
        triangles = vertices[faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 1e-10):
            raise ValueError("STL contains a degenerate zero-area triangle.")
        normals = normals / lengths[:, None]
        n_triangles = len(faces)

        fd, temp_path = tempfile.mkstemp(
            prefix=".pylcss_stl_", suffix=".stl", dir=os.path.dirname(filename) or "."
        )
        os.close(fd)
        try:
            with open(temp_path, 'wb') as f:
                # Header (80 bytes)
                header = b'Binary STL exported by PyLCSS (units: mm)' + b'\0' * 40
                f.write(header[:80])
                f.write(struct.pack('<I', n_triangles))

                for face, normal in zip(faces, normals, strict=True):
                    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                    f.write(struct.pack('<3f', *normal))
                    f.write(struct.pack('<3f', *v0))
                    f.write(struct.pack('<3f', *v1))
                    f.write(struct.pack('<3f', *v2))
                    f.write(struct.pack('<H', 0))
            if os.path.getsize(temp_path) != 84 + 50 * n_triangles:
                raise RuntimeError("Binary STL size verification failed.")
            os.replace(temp_path, filename)
            temp_path = None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


