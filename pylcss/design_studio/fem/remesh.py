# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM remesh node — surface mesh to volumetric mesh via Netgen (post-TopOpt)."""
import os
import tempfile
import logging

import numpy as np
import skfem

from pylcss.design_studio.core.base_node import CadQueryNode
from pylcss.design_studio.fem._helpers import suppress_output, OCCGeometry

logger = logging.getLogger(__name__)

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
        self.create_property('max_surface_faces', 2000, widget_type='int')
        self.create_property('allow_voxel_fallback', False, widget_type='bool')

    def run(self):
        """Convert recovered shape to volumetric mesh."""
        topopt_result = self.get_input_value('topopt_result', None)
        
        if topopt_result is None:
            logger.warning("RemeshNode: No TopOpt result provided")
            self.set_error("Connect a recovered surface or imported STL/OBJ mesh to Remesh Surface.")
            return None
        
        # Extract recovered shape from the full TopOpt result, while also
        # accepting a recovered_shape dict connected directly from the green
        # output port.  The node graph UI makes both connections plausible.
        recovered_shape = None
        if isinstance(topopt_result, dict):
            recovered_shape = topopt_result.get('recovered_shape', None)
            if recovered_shape is None and 'vertices' in topopt_result and 'faces' in topopt_result:
                recovered_shape = topopt_result
        
        if recovered_shape is None:
            self.set_error("Connect a TopOpt result, recovered_shape surface, or imported STL/OBJ mesh.")
            return None
        
        vertices = recovered_shape.get('vertices', None)
        faces = recovered_shape.get('faces', None)
        
        if vertices is None or faces is None:
            logger.warning("RemeshNode: Invalid recovered_shape format")
            self.set_error("The connected surface has no vertices or triangle faces.")
            return None

        raw_vertices = np.asarray(vertices, dtype=float)
        raw_faces = np.asarray(faces, dtype=int)
        prepared = self._prepare_surface_for_remesh(raw_vertices, raw_faces)
        if prepared is not None:
            vertices = np.asarray(prepared.vertices, dtype=float)
            faces = np.asarray(prepared.faces, dtype=int)
        
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
            requested_faces = int(self.get_property('max_surface_faces') or 0)
            if requested_faces > 0:
                tetgen_face_caps = []
                for cap in (
                    requested_faces,
                    max(2000, requested_faces * 2),
                    4000,
                    8000,
                    12000,
                ):
                    if cap > 0 and cap not in tetgen_face_caps:
                        tetgen_face_caps.append(cap)
            else:
                # A non-positive cap means "preserve the imported/recovered
                # surface"; never silently fall back to a coarse decimation.
                tetgen_face_caps = [0]
            for face_cap in tetgen_face_caps:
                tetgen_surface = self._prepare_surface_for_remesh(
                    raw_vertices,
                    raw_faces,
                    target_faces=face_cap,
                )
                if tetgen_surface is None:
                    continue
                mesh = self._remesh_via_tetgen_surface(
                    tetgen_surface.vertices,
                    tetgen_surface.faces,
                    effective_size,
                )
                if mesh is not None:
                    logger.info(
                        "RemeshNode: Created TetGen surface-conforming volume mesh with %d elements",
                        mesh.nelements,
                    )
                    return {
                        'mesh': mesh,
                        'shape': None,
                        'type': 'remesh',
                        'source': 'tetgen_surface',
                        'surface_faces': int(len(tetgen_surface.faces)),
                    }

            # Method 2: mesh the STL surface directly with Netgen.  This avoids
            # the slow/fragile triangle-to-CAD sewing path for TopOpt surfaces.
            mesh, solid = self._remesh_via_stl_geometry(
                vertices,
                faces,
                effective_size,
            )

            if mesh is not None:
                logger.info(
                    "RemeshNode: Created STL-derived volumetric mesh with %d elements",
                    mesh.nelements,
                )
                return {
                    'mesh': mesh,
                    'shape': solid,
                    'type': 'remesh',
                    'source': 'stl_surface',
                }

            if bool(self.get_property('allow_voxel_fallback')):
                mesh = self._remesh_via_voxel_fill(vertices, faces, effective_size)
                if mesh is not None:
                    logger.info(
                        "RemeshNode: Created opt-in voxel-filled fallback volume mesh with %d elements",
                        mesh.nelements,
                    )
                    return {
                        'mesh': mesh,
                        'shape': None,
                        'type': 'remesh',
                        'source': 'voxel_fill_surface',
                    }

            # Method 3: Create solid from surface mesh and mesh with Netgen.
            mesh, solid = self._remesh_via_solid(vertices, faces, effective_size)
            
            if mesh is not None:
                logger.info(f"RemeshNode: Created volumetric mesh with {mesh.nelements} elements")
                return {
                    'mesh': mesh,
                    'shape': solid,
                    'type': 'remesh'
                }
            
            # Do not fall back to scipy Delaunay for TopOpt surfaces.  That
            # method fills the convex hull and then guesses which tetrahedra
            # are inside from nearest-face normals; on organic marching-cubes
            # surfaces it can produce visibly broken volume meshes.
            self.set_error(
                "Remesh Surface could not create a surface-conforming solid "
                "tetra mesh from this STL. Enable the rough voxel fallback only "
                "for preview, or simplify/repair the recovered surface."
            )
            logger.warning(
                "RemeshNode: surface-conforming remesh failed; voxel fallback "
                "is disabled for FEA-grade workflows."
            )
            
        except Exception as e:
            logger.error(f"RemeshNode: Meshing failed: {e}")
            self.set_error(f"Remesh Surface failed: {e}")
        
        return None

    def _prepare_surface_for_remesh(self, vertices, faces, target_faces=None):
        """Repair and cap dense topology surfaces before volume meshing."""
        try:
            import trimesh

            verts = np.asarray(vertices, dtype=float)
            tris = np.asarray(faces, dtype=int)
            if verts.ndim != 2 or verts.shape[1] < 3 or tris.ndim != 2 or tris.shape[1] < 3:
                return None
            if len(verts) < 4 or len(tris) < 4:
                return None

            surface = trimesh.Trimesh(
                vertices=verts[:, :3],
                faces=tris[:, :3],
                process=bool(self.get_property('repair_surface')),
            )
            self._cleanup_surface(surface)
            if bool(self.get_property('close_holes')):
                try:
                    trimesh.repair.fill_holes(surface)
                except Exception:
                    pass
            try:
                trimesh.repair.fix_normals(surface)
            except Exception:
                pass

            face_limit = int(
                self.get_property('max_surface_faces') if target_faces is None else target_faces
                or 0
            )
            if face_limit > 0 and len(surface.faces) > face_limit:
                base_volume = 0.0
                try:
                    if surface.is_watertight:
                        base_volume = abs(float(surface.volume))
                except Exception:
                    base_volume = 0.0

                candidate_caps = [int(face_limit)]
                if base_volume > 1e-9:
                    for cap in (
                        max(face_limit + 1, face_limit * 2),
                        max(face_limit + 1, face_limit * 4),
                        4000,
                        8000,
                        12000,
                    ):
                        cap = min(int(cap), int(len(surface.faces)))
                        if cap > 0 and cap not in candidate_caps:
                            candidate_caps.append(cap)

                accepted = None
                best = None
                best_error = float('inf')
                for candidate_cap in candidate_caps:
                    try:
                        simplified = surface.simplify_quadric_decimation(
                            face_count=candidate_cap
                        )
                        if simplified is None or len(simplified.faces) < 4:
                            continue
                        self._cleanup_surface(simplified)
                        if bool(self.get_property('close_holes')):
                            try:
                                trimesh.repair.fill_holes(simplified)
                            except Exception:
                                pass
                        try:
                            trimesh.repair.fix_normals(simplified)
                        except Exception:
                            pass

                        volume_error = 0.0
                        if base_volume > 1e-9 and simplified.is_watertight:
                            volume_error = abs(abs(float(simplified.volume)) - base_volume) / base_volume
                        if volume_error < best_error:
                            best = simplified
                            best_error = volume_error
                        if base_volume <= 1e-9 or volume_error <= 0.015:
                            accepted = simplified
                            if candidate_cap != face_limit:
                                logger.info(
                                    "RemeshNode: raised surface cap from %d to %d "
                                    "to preserve STL volume (error %.2f%%).",
                                    face_limit,
                                    len(simplified.faces),
                                    100.0 * volume_error,
                                )
                            break
                    except Exception as exc:
                        logger.warning(
                            "RemeshNode: surface decimation to %d faces failed: %s",
                            candidate_cap,
                            exc,
                        )

                if accepted is not None:
                    surface = accepted
                elif best is not None and best_error <= 0.03:
                    logger.info(
                        "RemeshNode: using best available surface decimation "
                        "with %.2f%% volume error.",
                        100.0 * best_error,
                    )
                    surface = best
                elif best is not None:
                    logger.warning(
                        "RemeshNode: keeping repaired original STL surface; "
                        "requested cap %d would change volume by %.2f%%.",
                        face_limit,
                        100.0 * best_error,
                    )
            return surface if len(surface.vertices) >= 4 and len(surface.faces) >= 4 else None
        except Exception as exc:
            logger.warning("RemeshNode: surface preparation failed: %s", exc)
            return None

    @staticmethod
    def _cleanup_surface(surface):
        try:
            surface.remove_unreferenced_vertices()
            surface.merge_vertices()
            surface.remove_degenerate_faces()
            surface.remove_duplicate_faces()
        except Exception:
            pass

    def _remesh_via_tetgen_surface(self, vertices, faces, element_size):
        """Generate a true tetrahedral body mesh from a closed triangle surface."""
        try:
            import tetgen
            import trimesh
            from skfem import MeshTet

            surface = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=float)[:, :3],
                faces=np.asarray(faces, dtype=int)[:, :3],
                process=True,
            )
            self._cleanup_surface(surface)
            if bool(self.get_property('close_holes')):
                try:
                    trimesh.repair.fill_holes(surface)
                except Exception:
                    pass
            try:
                trimesh.repair.fix_normals(surface)
            except Exception:
                pass
            if len(surface.vertices) < 4 or len(surface.faces) < 4:
                return None
            if not surface.is_watertight:
                logger.warning(
                    "RemeshNode: TetGen requires a closed STL surface; surface is not watertight."
                )
                return None

            tet = tetgen.TetGen(
                np.asarray(surface.vertices, dtype=np.float64),
                np.asarray(surface.faces, dtype=np.int32),
            )
            maxvolume = max(float(element_size) ** 3 / 6.0, 1e-6)
            attempts = (
                {
                    "quality": True,
                    "nobisect": True,
                    "fixedvolume": True,
                    "minratio": 1.3,
                    "maxvolume": maxvolume,
                },
                {
                    "quality": True,
                    "nobisect": True,
                    "fixedvolume": True,
                    "minratio": 2.0,
                    "maxvolume": maxvolume,
                },
                {
                    "quality": True,
                    "fixedvolume": True,
                    "minratio": 1.5,
                    "maxvolume": maxvolume,
                },
                {
                    "quality": False,
                    "fixedvolume": True,
                    "maxvolume": maxvolume,
                },
            )
            last_error = None
            nodes = elements = None
            for attempt in attempts:
                try:
                    tet = tetgen.TetGen(
                        np.asarray(surface.vertices, dtype=np.float64),
                        np.asarray(surface.faces, dtype=np.int32),
                    )
                    nodes, elements, *_ = tet.tetrahedralize(
                        plc=True,
                        quiet=True,
                        nowarning=True,
                        **attempt,
                    )
                    break
                except Exception as exc:
                    last_error = exc
            if nodes is None or elements is None:
                if last_error is not None:
                    logger.warning("RemeshNode: TetGen quality meshing failed: %s", last_error)
                return None

            pts = np.asarray(nodes, dtype=float)
            tet_arr = np.asarray(elements, dtype=int)
            if pts.ndim != 2 or pts.shape[1] < 3 or tet_arr.ndim != 2 or tet_arr.shape[1] < 4:
                return None
            keep = np.ones(len(tet_arr), dtype=bool)
            for idx, row in enumerate(tet_arr):
                p0, p1, p2, p3 = pts[row[:4]]
                vol6 = float(np.linalg.det(np.vstack([p1 - p0, p2 - p0, p3 - p0])))
                if vol6 < 0.0:
                    row[1], row[2] = row[2], row[1]
                    vol6 = -vol6
                if vol6 <= 1e-8:
                    keep[idx] = False
            tet_arr = tet_arr[keep]
            if len(tet_arr) < 1:
                return None
            used, inverse = np.unique(tet_arr[:, :4].reshape(-1), return_inverse=True)
            pts = pts[used]
            tet_arr = inverse.reshape((-1, 4))
            return MeshTet(pts[:, :3].T, tet_arr[:, :4].T)
        except Exception as exc:
            logger.warning("RemeshNode: TetGen remesh failed: %s", exc)
            return None

    def _remesh_via_stl_geometry(self, vertices, faces, element_size):
        """Volume-mesh a closed STL-like triangle surface using Netgen-STL.

        This is the preferred path for topology-optimization output and imported
        STL files because it does not first convert triangles into an OCC B-rep.
        The input must still be a reasonably closed, consistently oriented
        surface; if Netgen cannot create a volume mesh, the caller falls back to
        the older sewing path.
        """
        stl_path = None
        msh_path = None
        try:
            import trimesh
            from netgen.stl import STLGeometry

            verts = np.asarray(vertices, dtype=float)
            tris = np.asarray(faces, dtype=int)
            if verts.ndim != 2 or verts.shape[1] < 3 or tris.ndim != 2 or tris.shape[1] < 3:
                return None, None
            if len(verts) < 4 or len(tris) < 4:
                return None, None

            surface = trimesh.Trimesh(
                vertices=verts[:, :3],
                faces=tris[:, :3],
                process=bool(self.get_property('repair_surface')),
            )
            self._cleanup_surface(surface)
            if bool(self.get_property('close_holes')):
                try:
                    trimesh.repair.fill_holes(surface)
                except Exception:
                    pass
            try:
                trimesh.repair.fix_normals(surface)
            except Exception:
                pass

            if len(surface.vertices) < 4 or len(surface.faces) < 4:
                return None, None
            if not surface.is_watertight:
                logger.warning(
                    "RemeshNode: STL surface is not watertight; Netgen STL "
                    "volume meshing may fail."
                )

            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as stl_file:
                stl_path = stl_file.name
            msh_path = stl_path + '.msh'
            surface.export(stl_path, file_type='stl')

            with suppress_output():
                geo = STLGeometry(stl_path)
                ngmesh = geo.GenerateMesh(maxh=float(element_size))
                ngmesh.Export(msh_path, 'Gmsh2 Format')

            mesh = skfem.MeshTet.load(msh_path)
            return mesh, None
        except Exception as exc:
            logger.warning("RemeshNode: Netgen STL remesh failed: %s", exc)
            return None, None
        finally:
            for path in (stl_path, msh_path):
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

    def _remesh_via_voxel_fill(self, vertices, faces, element_size):
        """Fallback volume mesh by voxel-filling a watertight STL surface.

        This is intentionally a validation fallback: it gives CalculiX and
        OpenRadioss a conservative tetra volume when a complex recovered STL is
        too irregular for Netgen's STL front-end.
        """
        try:
            import trimesh
            from skfem import MeshTet

            surface = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=float)[:, :3],
                faces=np.asarray(faces, dtype=int)[:, :3],
                process=True,
            )
            self._cleanup_surface(surface)
            try:
                trimesh.repair.fix_normals(surface)
            except Exception:
                pass
            if len(surface.vertices) < 4 or len(surface.faces) < 4:
                return None

            mins = np.asarray(surface.bounds[0], dtype=float)
            maxs = np.asarray(surface.bounds[1], dtype=float)
            span = np.maximum(maxs - mins, 1e-9)
            min_span = float(np.min(span))
            step = min(float(element_size), max(min_span / 4.0, 1.0))
            counts = np.maximum(np.ceil(span / max(step, 1e-9)).astype(int), 1)
            max_cells = 25000
            cell_count = int(np.prod(counts))
            if cell_count > max_cells:
                scale = (cell_count / float(max_cells)) ** (1.0 / 3.0)
                counts = np.maximum(np.ceil(counts / scale).astype(int), 1)
            cell = span / counts.astype(float)

            ix, iy, iz = np.meshgrid(
                np.arange(counts[0], dtype=int),
                np.arange(counts[1], dtype=int),
                np.arange(counts[2], dtype=int),
                indexing='ij',
            )
            indices = np.column_stack([ix.ravel(), iy.ravel(), iz.ravel()])
            centers = mins + (indices.astype(float) + 0.5) * cell
            try:
                inside = surface.contains(centers)
            except Exception:
                from trimesh.proximity import signed_distance
                inside = signed_distance(surface, centers) >= 0.0
            active = indices[np.asarray(inside, dtype=bool)]
            if active.size == 0:
                from trimesh.proximity import signed_distance
                active = indices[signed_distance(surface, centers) >= -0.15 * float(np.mean(cell))]
            if active.size == 0:
                return None

            node_ids = {}
            points = []

            def point_id(i, j, k):
                key = (int(i), int(j), int(k))
                existing = node_ids.get(key)
                if existing is not None:
                    return existing
                coord = mins + np.asarray(key, dtype=float) * cell
                node_ids[key] = len(points)
                points.append(coord)
                return node_ids[key]

            local_corners = (
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
            )
            local_tets = (
                (0, 1, 3, 7),
                (0, 3, 2, 7),
                (0, 2, 6, 7),
                (0, 6, 4, 7),
                (0, 4, 5, 7),
                (0, 5, 1, 7),
            )
            tets = []
            for i, j, k in active:
                corners = [point_id(i + dx, j + dy, k + dz) for dx, dy, dz in local_corners]
                for tet in local_tets:
                    tets.append([corners[idx] for idx in tet])

            pts = np.asarray(points, dtype=float)
            tet_arr = np.asarray(tets, dtype=int)
            if len(pts) < 4 or len(tet_arr) < 1:
                return None
            for row in tet_arr:
                p0, p1, p2, p3 = pts[row]
                vol6 = float(np.linalg.det(np.vstack([p1 - p0, p2 - p0, p3 - p0])))
                if vol6 < 0.0:
                    row[1], row[2] = row[2], row[1]

            return MeshTet(pts.T, tet_arr.T)
        except Exception as exc:
            logger.warning("RemeshNode: voxel-fill fallback failed: %s", exc)
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
