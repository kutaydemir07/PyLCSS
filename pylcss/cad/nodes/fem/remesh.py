# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM remesh node — surface mesh to volumetric mesh via Netgen (post-TopOpt)."""
import numpy as np
import logging
import os
import tempfile
import sys
import contextlib
from scipy.spatial import cKDTree
import skfem
from skfem import *
from skfem.helpers import sym_grad, ddot, trace
try:
    from simpleeval import simple_eval
except ImportError:
    simple_eval = None
from pylcss.config import simulation_config
from pylcss.cad.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)
from pylcss.cad.nodes.fem._helpers import (
    lam_lame, tr, suppress_output, OCCGeometry,
    _find_matching_boundary_facets,
)

class RemeshNode(CadQueryNode):
    """
    Remesh Node - Converts surface mesh (from TopOpt) to volumetric tetrahedral mesh.
    
    This node bridges TopOpt → ShapeOpt workflow by taking the recovered shape
    (surface triangles) and creating a new volumetric mesh suitable for FEA.
    """
    __identifier__ = 'com.cad.sim.remesh'
    NODE_NAME = 'Remesh Surface'

    def __init__(self):
        super().__init__()
        # Input: TopOpt result containing recovered_shape
        self.add_input('topopt_result', color=(200, 100, 200))
        
        # Output: Volumetric mesh
        self.add_output('mesh', color=(200, 100, 200))
        self.add_output('shape', color=(100, 255, 100))  # CadQuery solid for visualization
        
        # Mesh quality settings
        self.create_property('element_size', 3.0, widget_type='float')
        self.create_property('mesh_quality', 'Medium', widget_type='combo',
                             items=['Coarse', 'Medium', 'Fine', 'Very Fine'])
        
        # Surface repair options
        self.create_property('repair_surface', True, widget_type='bool')
        self.create_property('close_holes', True, widget_type='bool')

    def run(self):
        """Convert recovered shape to volumetric mesh."""
        topopt_result = self.get_input_value('topopt_result', None)
        
        if topopt_result is None:
            logger.warning("RemeshNode: No TopOpt result provided")
            return None
        
        # Extract recovered shape from TopOpt result
        recovered_shape = None
        if isinstance(topopt_result, dict):
            recovered_shape = topopt_result.get('recovered_shape', None)
        
        if recovered_shape is None:
            self.set_warning("String conditions cannot be visualized pre-solve.")
            return None
        
        vertices = recovered_shape.get('vertices', None)
        faces = recovered_shape.get('faces', None)
        
        if vertices is None or faces is None:
            logger.warning("RemeshNode: Invalid recovered_shape format")
            return None
        
        logger.info(f"RemeshNode: Processing surface with {len(vertices)} vertices, {len(faces)} faces")
        
        element_size = self.get_property('element_size')
        quality = self.get_property('mesh_quality')
        
        # Adjust element size based on quality setting
        quality_multipliers = {
            'Coarse': 2.0,
            'Medium': 1.0,
            'Fine': 0.5,
            'Very Fine': 0.25
        }
        effective_size = element_size * quality_multipliers.get(quality, 1.0)
        
        try:
            # Method 1: Create solid from surface mesh and mesh with Netgen
            mesh, solid = self._remesh_via_solid(vertices, faces, effective_size)
            
            if mesh is not None:
                logger.info(f"RemeshNode: Created volumetric mesh with {mesh.nelements} elements")
                return {
                    'mesh': mesh,
                    'shape': solid,
                    'type': 'remesh'
                }
            
            # Method 2 fallback: Direct tetrahedral mesh from surface
            mesh = self._remesh_direct(vertices, faces, effective_size)
            if mesh is not None:
                logger.info(f"RemeshNode: Created mesh via direct method with {mesh.nelements} elements")
                return {
                    'mesh': mesh,
                    'shape': None,
                    'type': 'remesh'
                }
            
        except Exception as e:
            logger.error(f"RemeshNode: Meshing failed: {e}")
        
        return None
    
    def _remesh_via_solid(self, vertices, faces, element_size):
        """Create volumetric mesh by first creating a solid from surface.

        .. warning::
            This method passes raw marching-cubes output (highly triangulated,
            non-manifold triangles) to OpenCASCADE's ``BRepBuilderAPI_Sewing`` /
            ``ShapeFix_Solid`` pipeline.  OCC sewing is designed for CAD B-Rep
            surfaces, *not* dense isosurface meshes.  Common failure modes on
            marching-cubes output include:

            * Sewing never terminates (exponential cost at >5 k triangles).
            * Shell fails to close → ``BRepBuilderAPI_MakeSolid`` returns Nothing.
            * ``ShapeFix_Solid`` silently produces an incorrect orientated solid.

            Recommended robust alternative
            --------------------------------
            1. Reduce face count *before* sewing:
               ``pyvista`` / ``trimesh`` Laplacian smoothing + quadric-decimation
               typically reduce a 50 k-face marching-cubes mesh to <2 k faces
               while preserving topology.
            2. Skip B-Rep altogether and remesh the smoothed STL directly with
               ``tetgen`` or ``fTetWild`` to obtain a quality tetrahedral mesh
               suitable for skfem.
        """
        # Performance guard: very large meshes are slow to sew into B-Rep.
        # Raise the limit to 20 000 faces (typical TopOpt output) with a warning.
        if len(faces) > 20000:
            logger.warning(
                f"RemeshNode: Mesh has {len(faces)} faces (> 20 000). "
                "OCC sewing on marching-cubes output at this density is likely to "
                "hang or produce a broken solid. Skipping solid conversion. "
                "Consider using pyvista/trimesh decimation + tetgen for robust remeshing."
            )
            return None, None
        if len(faces) > 5000:
            logger.warning(
                f"RemeshNode: Large marching-cubes mesh ({len(faces)} faces). "
                "OCC sewing may be slow or fail. "
                "Laplacian smoothing + decimation (pyvista/trimesh) is strongly recommended "
                "before B-Rep conversion."
            )
            
        mesh = None
        solid = None
        
        # Helper to create solid from faces
        try:
            import cadquery as cq
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
            from OCP.BRep import BRep_Builder
            from OCP.TopoDS import TopoDS_Shell, TopoDS_Compound, TopoDS_Solid
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.gp import gp_Pnt
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
            from OCP.ShapeFix import ShapeFix_Solid, ShapeFix_Shell
            
            logger.info("RemeshNode: Attempting surface sewing to create solid...")
            
            # Build shell from triangular faces
            sew = BRepBuilderAPI_Sewing(1e-4)  # Tighter tolerance
            
            for face in faces:
                try:
                    # Get triangle vertices
                    pts = [gp_Pnt(float(vertices[idx, 0]), 
                                  float(vertices[idx, 1]), 
                                  float(vertices[idx, 2])) for idx in face]
                    
                    # Create triangular face
                    poly = BRepBuilderAPI_MakePolygon(pts[0], pts[1], pts[2], True)
                    if not poly.IsDone(): continue
                        
                    wire = poly.Wire()
                    face_builder = BRepBuilderAPI_MakeFace(wire, True)
                    if face_builder.IsDone():
                        sew.Add(face_builder.Face())
                except Exception:
                    continue
            
            sew.Perform()
            sewed_shape = sew.SewedShape()

            # Extract the shell to pass to ShapeFix_Shell.
            # BRepBuilderAPI_Sewing may return a TopoDS_Compound when the
            # input triangles form more than one disconnected patch, or simply
            # because OCC wraps singletons in a Compound.  Attempting a direct
            # downcast TopoDS_Shell(compound) causes a C++ type-assertion
            # exception.  We must use TopExp_Explorer to pull the first shell.
            from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_SHELL
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopoDS import topods

            if sewed_shape.ShapeType() == TopAbs_COMPOUND:
                explorer = TopExp_Explorer(sewed_shape, TopAbs_SHELL)
                if explorer.More():
                    shell_shape = explorer.Current()
                else:
                    raise RuntimeError(
                        "RemeshNode: sewing produced a Compound with no Shell — "
                        "mesh may be non-manifold or have open boundaries."
                    )
            else:
                shell_shape = sewed_shape

            # Try to fix shell
            fixer = ShapeFix_Shell(topods.Shell(shell_shape))
            fixer.Perform()
            shell = fixer.Shell()
            
            # Make solid
            solid_builder = BRepBuilderAPI_MakeSolid()
            solid_builder.Add(shell)
            
            if solid_builder.IsDone():
                occ_solid = solid_builder.Solid()
                
                # Fix solid orientation/volume
                fixer_sol = ShapeFix_Solid(occ_solid)
                fixer_sol.Perform()
                occ_solid = fixer_sol.Solid()
                
                solid = cq.Workplane().add(cq.Shape(occ_solid))
                
                # Mesh with Netgen if available
                if OCCGeometry is not None:
                    try:
                        geo = OCCGeometry(occ_solid)
                        ngmesh = geo.GenerateMesh(maxh=element_size)
                        
                        # Export/Import cycle for skfem
                        with tempfile.NamedTemporaryFile(suffix='.msh', delete=False) as f:
                            ngmesh.Export(f.name, 'Gmsh2 Format')
                            f.close()
                            mesh = skfem.MeshTet.load(f.name)
                            os.unlink(f.name)
                            
                    except Exception as e:
                        logger.warning(f"RemeshNode: Netgen meshing failed: {e}")
            else:
                logger.warning("RemeshNode: Failed to close shell into solid")
                
        except Exception as e:
            logger.warning(f"RemeshNode: Solid creation failed: {e}")
        
        return mesh, solid
    
    def _remesh_direct(self, vertices, faces, element_size):
        """Direct tetrahedral meshing using scipy Delaunay."""
        try:
            from scipy.spatial import Delaunay
            
            # Create 3D Delaunay triangulation of the vertices
            # This creates tetrahedra filling the convex hull
            tri = Delaunay(vertices)
            
            # Filter tetrahedra to keep only those inside the surface
            # Use point-in-mesh test based on face normals + KDTree
            centroids = vertices[tri.simplices].mean(axis=1)
            
            # Simple inside test: keep tetrahedra whose centroids are near original surface
            from scipy.spatial import cKDTree
            face_centers = vertices[faces].mean(axis=1)
            tree = cKDTree(face_centers)
            
            # For each tetrahedron centroid, find distance to nearest face center
            distances, indices = tree.query(centroids)
            
            # Estimate typical edge length from sample faces
            edge_lengths = []
            for face in faces[:min(100, len(faces))]:
                pts = vertices[face]
                vals = [np.linalg.norm(pts[i] - pts[(i+1)%3]) for i in range(3)]
                edge_lengths.extend(vals)
            avg_edge = np.mean(edge_lengths) if edge_lengths else element_size
            
            # Heuristic: keep tets that are "close enough" to the surface shell
            # Ideally we would do a winding number check, but that's expensive in pure python/scipy
            # A threshold of avg_edge * 1.5 usually keeps the bulk without too many outliers
            # Also, check if centroid is "behind" the nearest face (dot product of normal)
            
            valid_mask = np.zeros(len(centroids), dtype=bool)
            
            # Precompute face normals
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            normals = np.cross(v1 - v0, v2 - v0)
            norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10
            normals /= norms
            
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist > avg_edge * 3.0: # Too far, definitely outside
                    continue
                
                # Check direction (is centroid 'inside' relative to nearest face?)
                # Vector from face center to centroid
                vec = centroids[i] - face_centers[idx]
                val = np.dot(vec, normals[idx])
                
                # If dot product is negative, it's on the 'back' side of the face (inside)
                # Or if it's very close (within tolerance)
                if val < 0.1 * avg_edge: 
                    valid_mask[i] = True
            
            valid_simplices = tri.simplices[valid_mask]
            
            if len(valid_simplices) > 0:
                mesh = skfem.MeshTet(vertices.T, valid_simplices.T)
                return mesh
                
        except Exception as e:
            logger.debug(f"RemeshNode: Direct method failed: {e}")
        
        return None


