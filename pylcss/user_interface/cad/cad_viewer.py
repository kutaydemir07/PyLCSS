# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import vtk
import numpy as np
from PySide6 import QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class CQ3DViewer(QtWidgets.QWidget):
    """
    Professional 3D Viewer for CadQuery using VTK.
    Embeds directly into PySide6 layouts.
    """
    def __init__(self, parent=None):
        super(CQ3DViewer, self).__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # VTK Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtkWidget)

        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)  # Dark Gray Background
        
        # Scalar Bar (Legend)
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetOrientationToVertical()
        self.scalar_bar.SetWidth(0.1)
        self.scalar_bar.SetHeight(0.8)
        self.scalar_bar.SetPosition(0.85, 0.1)
        self.scalar_bar.VisibilityOff()
        self.renderer.AddActor(self.scalar_bar)
        
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Initialize axes (XYZ arrows)
        axes = vtk.vtkAxesActor()
        self.marker_widget = vtk.vtkOrientationMarkerWidget()
        self.marker_widget.SetOrientationMarker(axes)
        self.marker_widget.SetInteractor(self.interactor)
        self.marker_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.marker_widget.SetEnabled(1)

        # State
        self.current_actor = None
        self.actors = []  # List of all active actors
        self.interactor.Initialize()
        self.interactor.Start()

    def clear(self):
        """Clear the viewer and release memory."""
        # Clear legacy single actor if present
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            if self.current_actor.GetMapper():
                self.current_actor.GetMapper().RemoveAllInputConnections(0)
            self.current_actor = None
            
        # Clear all actors in the list
        for actor in self.actors:
            self.renderer.RemoveActor(actor)
            if actor.GetMapper():
                actor.GetMapper().RemoveAllInputConnections(0)
        self.actors = []
            
        self.scalar_bar.VisibilityOff()
        self.vtkWidget.GetRenderWindow().Render()

    def _update_scalar_bar(self, title, min_val, max_val, lut=None):
        """Update and show the scalar bar."""
        self.scalar_bar.SetTitle(title)
        self.scalar_bar.SetNumberOfLabels(5)
        
        if lut:
            self.scalar_bar.SetLookupTable(lut)
        elif self.current_actor and self.current_actor.GetMapper():
            self.scalar_bar.SetLookupTable(self.current_actor.GetMapper().GetLookupTable())
            
        self.scalar_bar.VisibilityOn()

    def render_sketch(self, sketch):
        """
        Render a 2D sketch (CadQuery Workplane with 2D geometry) as polylines.
        Works with sketches that have wires but no 3D solid.
        """
        if sketch is None:
            return
            
        self.clear()
        
        # Try to extract edges from the sketch
        edges = []
        try:
            # Method 1: CadQuery Workplane with pending wires
            if hasattr(sketch, 'ctx') and hasattr(sketch.ctx, 'pendingWires'):
                try:
                    wires = sketch.ctx.pendingWires
                    if wires:
                        for wire in wires:
                            if hasattr(wire, 'Edges'):
                                edges.extend(wire.Edges())
                except Exception:
                    pass
            
            # Method 2: Try to get edges directly from the unwrapped shape
            if not edges:
                shape = sketch
                if hasattr(sketch, 'val'):
                    try:
                        shape = sketch.val()
                    except:
                        pass
                
                # Check for Edges() method directly on the shape
                if hasattr(shape, 'Edges'):
                    try:
                        edge_list = shape.Edges()
                        if edge_list:
                            edges = edge_list
                    except Exception:
                        pass
            
            # Method 3: Try the CadQuery .edges() API
            if not edges and hasattr(sketch, 'edges'):
                try:
                    edge_objects = sketch.edges().vals()
                    if edge_objects:
                        edges = edge_objects
                except Exception:
                    pass
            
            # Method 4: Check for wires on the shape
            if not edges:
                shape = sketch
                if hasattr(sketch, 'val'):
                    try:
                        shape = sketch.val()
                    except:
                        pass
                        
                if hasattr(shape, 'Wires'):
                    try:
                        wires = shape.Wires()
                        for wire in wires:
                            if hasattr(wire, 'Edges'):
                                edges.extend(wire.Edges())
                        if edges:
                            pass
                    except Exception:
                        pass
            
            # Method 5: Try _edges attribute (fallback)
            if not edges and hasattr(sketch, '_edges'):
                edges = sketch._edges
                
        except Exception:
            return
            
        if not edges:
            return
            
        # Create VTK points and lines
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        point_id = 0
        
        for edge in edges:
            try:
                pts_extracted = []
                
                # Method 1: Use OCCT curve sampling (best for circles, arcs, splines)
                if hasattr(edge, 'wrapped'):
                    try:
                        from OCP.BRepAdaptor import BRepAdaptor_Curve
                        from OCP.GCPnts import GCPnts_UniformAbscissa
                        
                        curve = BRepAdaptor_Curve(edge.wrapped)
                        # Sample 30 points along the edge for smooth curves
                        sampler = GCPnts_UniformAbscissa(curve, 30)
                        
                        if sampler.NbPoints() > 0:
                            for i in range(1, sampler.NbPoints() + 1):
                                p = curve.Value(sampler.Parameter(i))
                                pts_extracted.append((p.X(), p.Y(), p.Z()))
                    except Exception:
                        pass
                
                # Method 2: CadQuery edge discretization (if OCCT didn't work)
                if not pts_extracted and hasattr(edge, 'discretize'):
                    try:
                        pts_extracted = edge.discretize(30)  # More points for smooth curves
                    except Exception:
                        pass
                
                # Method 3: Simple start/end for straight lines
                if not pts_extracted and hasattr(edge, 'startPoint') and hasattr(edge, 'endPoint'):
                    sp = edge.startPoint()
                    ep = edge.endPoint()
                    # Check if start != end (not a closed curve with only 2 points)
                    if abs(sp.x - ep.x) > 1e-6 or abs(sp.y - ep.y) > 1e-6 or abs(sp.z - ep.z) > 1e-6:
                        pts_extracted = [(sp.x, sp.y, sp.z), (ep.x, ep.y, ep.z)]
                
                # Add points to VTK
                if pts_extracted and len(pts_extracted) >= 2:
                    start_id = point_id
                    for pt in pts_extracted:
                        if hasattr(pt, 'x'):
                            points.InsertNextPoint(pt.x, pt.y, pt.z if hasattr(pt, 'z') else 0)
                        else:
                            points.InsertNextPoint(pt[0], pt[1], pt[2] if len(pt) > 2 else 0)
                        point_id += 1
                    
                    # Create polyline from these points
                    for i in range(start_id, point_id - 1):
                        line = vtk.vtkLine()
                        line.GetPointIds().SetId(0, i)
                        line.GetPointIds().SetId(1, i + 1)
                        lines.InsertNextCell(line)
                    
                    # Check if this is a closed curve (circle, ellipse) and close it
                    if len(pts_extracted) > 2:
                        first_pt = pts_extracted[0]
                        last_pt = pts_extracted[-1]
                        if hasattr(first_pt, 'x'):
                            dist = abs(first_pt.x - last_pt.x) + abs(first_pt.y - last_pt.y)
                        else:
                            dist = abs(first_pt[0] - last_pt[0]) + abs(first_pt[1] - last_pt[1])
                        
                        # If first and last points are close, it's likely a closed curve - don't add extra line
                        # The curve is already visually closed
                    
            except Exception:
                continue
        
        if point_id == 0:
            return  # No points extracted
            
        # Create polydata
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Style for sketch (bright cyan lines)
        actor.GetProperty().SetColor(0.0, 0.9, 0.9)  # Cyan
        actor.GetProperty().SetLineWidth(2.0)
        
        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)
        
        # Set camera to top-down view for 2D sketch
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 0, 100)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
        
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()


    def render_shape(self, shape):
        """
        Accepts a CadQuery Workplane or Shape, tessellates it, and renders it.
        """
        self.scalar_bar.VisibilityOff()
        if shape is None:
            return

        # CHANGE: Use the robust clear method first
        self.clear()

        # 2. Extract geometry from CadQuery
        # Handle different CQ types (Workplane vs Shape vs Assembly)
        topo_shape = shape

        # If it's an assembly, convert to compound
        if hasattr(shape, 'toCompound'):
            try:
                topo_shape = shape.toCompound()
            except Exception:
                return

        # If it's a wrapper Workplane with .val(), try to unwrap
        try:
            if hasattr(topo_shape, 'val'):
                topo_shape = topo_shape.val()
        except Exception:
            pass

        # If it's a container (list/tuple) try to find a tessellatable item
        if not hasattr(topo_shape, 'tessellate'):
            if isinstance(topo_shape, (list, tuple)) and topo_shape:
                found = None
                for item in topo_shape:
                    if hasattr(item, 'tessellate'):
                        found = item
                        break
                    if hasattr(item, 'val'):
                        try:
                            v = item.val()
                            if hasattr(v, 'tessellate'):
                                found = v
                                break
                        except Exception:
                            continue
                if found is not None:
                    topo_shape = found

        # Some CQ objects expose .objects with shapes inside
        if not hasattr(topo_shape, 'tessellate') and hasattr(topo_shape, 'objects'):
            try:
                for obj in getattr(topo_shape, 'objects', []):
                    if hasattr(obj, 'val'):
                        v = obj.val()
                        if hasattr(v, 'tessellate'):
                            topo_shape = v
                            break
                    if hasattr(obj, 'tessellate'):
                        topo_shape = obj
                        break
            except Exception:
                pass

        # Safety check: ensure it's a valid shape
        if not hasattr(topo_shape, 'tessellate'):
            # Unable to tessellate - try as 2D sketch instead
            self.render_sketch(shape)
            return

        # 3. Tessellate (convert curved CAD surfaces to triangles for viewing)
        try:
            # Fast rendering with coarser tessellation (0.1 tolerance)
            triangulation = topo_shape.tessellate(tolerance=0.1, angularTolerance=0.2)
            if isinstance(triangulation, dict):
                verts = triangulation.get('vertices') or triangulation.get('verts')
                triangles = triangulation.get('triangles') or triangulation.get('faces')
            else:
                verts, triangles = triangulation[0], triangulation[1]
        except Exception as e:
            # Tessellation failed - try as 2D sketch instead
            self.render_sketch(shape)
            return

        # Check if tessellation produced any geometry
        if not verts or len(verts) == 0:
            # No vertices - this is likely 2D geometry, try render_sketch
            self.render_sketch(shape)
            return

        # 4. Create VTK Data
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()

        # Add Vertices
        for v in verts:
            points.InsertNextPoint(v.x, v.y, v.z)

        # Add Triangles
        for t in triangles:
            polys.InsertNextCell(3)
            polys.InsertCellPoint(t[0])
            polys.InsertCellPoint(t[1])
            polys.InsertCellPoint(t[2])

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(polys)
        
        # Calculate Normals for smooth shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly_data)
        normals.Update()

        # 5. Mapper and Actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Styling (Metallic Blue/Grey)
        actor.GetProperty().SetColor(0.7, 0.75, 0.8) 
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(20)

        # 6. Add to scene
        self.renderer.AddActor(actor)
        self.current_actor = actor
        
        # Reset camera to fit object
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def _add_arrow(self, start, vector, color=(1, 1, 0), scale=1.0):
        """Add a 3D arrow to the scene representing a vector."""
        arrow_source = vtk.vtkArrowSource()
        
        # Determine logical length from vector magnitude
        length = np.linalg.norm(vector)
        if length < 1e-9:
            return
            
        # For visualization, we might want to normalize the size or use the real magnitude
        # Let's use a fixed visual size scaled by 'scale' but oriented correctly
        # actually, let's just draw it with length = scale * 10 or something
        # Better: Draw it with actual length if it fits, or normalized length.
        # Let's assume vector is the actual force vector (e.g. 1000N). That's too huge for global coords (20mm box).
        # We should normalize it to a reasonable visual size, e.g. 20% of bounding box.
        # For now, let's just make it length 5.0 (arbitrary visual size)
        visual_length = 5.0 * scale
        
        normalized_vector = vector / length
        
        transform = vtk.vtkTransform()
        transform.Translate(start[0], start[1], start[2])
        
        # VTK arrow default points along X+. orientation logic:
        default_dir = np.array([1.0, 0.0, 0.0])
        axis = np.cross(default_dir, normalized_vector)
        angle_rad = np.arccos(np.clip(np.dot(default_dir, normalized_vector), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        if np.linalg.norm(axis) > 1e-6:
            transform.RotateWXYZ(angle_deg, axis)
        elif np.dot(default_dir, normalized_vector) < 0:
            transform.RotateWXYZ(180, [0, 1, 0])
            
        transform.Scale(visual_length, visual_length, visual_length)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(arrow_source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetUserTransform(transform)
        actor.GetProperty().SetColor(color)
        
        self.renderer.AddActor(actor)
        self.actors.append(actor)

    def _add_cube_marker(self, pos, color=(1, 0, 0), size=1.0):
        """Add a cube marker (e.g. for constraints)."""
        source = vtk.vtkCubeSource()
        source.SetCenter(pos[0], pos[1], pos[2])
        source.SetXLength(size)
        source.SetYLength(size)
        source.SetZLength(size)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        
        self.renderer.AddActor(actor)
        self.actors.append(actor)

    def update_simulation_field(self, mesh, values, field_name="Density"):
        """Update scalar field on existing mesh for real-time visualization."""
        data = {
            'mesh': mesh,
            'visualization_mode': field_name
        }
        if field_name == 'Density':
            data['density'] = values
        elif field_name == 'Von Mises Stress':
            data['stress'] = values
            
        self.render_simulation(data)

    def render_simulation(self, data):
        """
        Render simulation results (Mesh or FEA Result).
        """
        if data is None:
            return

        # 1. Clean up previous render
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None

        # Check if it's a Mesh object or Result dict
        mesh = None
        displacement = None
        density = None
        stress = None
        visualization_mode = 'Von Mises Stress'
        density_cutoff = 0.5
        
        # Detect skfem Mesh
        if hasattr(data, 'p') and hasattr(data, 't'):
            mesh = data
        # Detect Result Dict
        elif isinstance(data, dict) and 'mesh' in data:
            mesh = data['mesh']
            if 'displacement' in data:
                displacement = data['displacement']
            if 'density' in data:
                density = data['density']
            if 'stress' in data:
                stress = data['stress']
            if 'visualization_mode' in data:
                visualization_mode = data['visualization_mode']
            if 'density_cutoff' in data:
                density_cutoff = float(data['density_cutoff'])
        
        if mesh is None:
            return


        # 2. Create VTK Unstructured Grid
        points = vtk.vtkPoints()
        grid = vtk.vtkUnstructuredGrid()
        
        # Add Points (Nodes)
        # mesh.p is (3, N)
        pts = mesh.p
        n_points = pts.shape[1]
        
        # Apply displacement if available
        if displacement is not None:
            # displacement is flat vector [u1, v1, w1, u2, v2, w2, ...]
            # Reshape to (N, 3) or (3, N)
            if len(displacement) == 3 * n_points:
                # Assuming Fortran order (component major)
                disp_3n = displacement.reshape((3, n_points), order='F')
                pts = pts + disp_3n * 1.0 # Scale factor
            
        for i in range(n_points):
            points.InsertNextPoint(pts[0, i], pts[1, i], pts[2, i])
            
        grid.SetPoints(points)
        
        # Add Elements (Tetrahedra)
        # mesh.t is (4, M)
        tets = mesh.t
        n_tets = tets.shape[1]
        
        # Add Density Data if available
        if density is not None:
            density_array = vtk.vtkFloatArray()
            density_array.SetName("Density")
            for d in density:
                density_array.InsertNextValue(float(d))
            grid.GetCellData().SetScalars(density_array)

        # Add Stress/Displacement Scalars to Grid (so they persist through thresholding)
        if stress is not None:
            s_array = vtk.vtkFloatArray()
            s_array.SetName("VonMises")
            for s in stress:
                s_array.InsertNextValue(float(s))
            grid.GetPointData().AddArray(s_array)
            if visualization_mode == 'Von Mises Stress':
                grid.GetPointData().SetActiveScalars("VonMises")

        if displacement is not None:
            if len(displacement) == 3 * n_points:
                disp_3n = displacement.reshape((3, n_points), order='F')
                mag = np.linalg.norm(disp_3n, axis=0)
                mag_array = vtk.vtkFloatArray()
                mag_array.SetName("Displacement")
                for m in mag:
                    mag_array.InsertNextValue(m)
                grid.GetPointData().AddArray(mag_array)
                if visualization_mode == 'Displacement':
                    grid.GetPointData().SetActiveScalars("Displacement")
        
        for i in range(n_tets):
            tet = vtk.vtkTetra()
            for j in range(4):
                tet.GetPointIds().SetId(j, tets[j, i])
            grid.InsertNextCell(tet.GetCellType(), tet.GetPointIds())
            
        # 3. Mapper and Actor
        mapper = vtk.vtkDataSetMapper()
        
        if density is not None:
            # Thresholding for Topology Optimization
            # Use user-provided cutoff when available; fall back to a mild threshold
            cutoff = float(np.clip(density_cutoff, 0.05, 0.95))
            lower = max(0.01, cutoff)
            upper = 1.1

            threshold = vtk.vtkThreshold()
            threshold.SetInputData(grid)
            threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Density")
            
            # Fix for VTK 9.x compatibility
            threshold.SetLowerThreshold(lower)
            threshold.SetUpperThreshold(upper)
            # 0 is usually THRESHOLD_BETWEEN
            threshold.SetThresholdFunction(getattr(vtk.vtkThreshold, 'THRESHOLD_BETWEEN', 0))
            
            # IMPORTANT: Pass point arrays (like VonMises stress) through the threshold filter
            # Without this, stress data is lost when thresholding by density
            try:
                threshold.SetPassPointArrays(True)
            except AttributeError:
                # Older VTK versions may not have this method
                pass
            
            threshold.Update()

            threshold_output = threshold.GetOutput()

            # If nothing passes the cutoff, relax it once to avoid showing the full solid
            if threshold_output.GetNumberOfCells() == 0 and len(density) > 0:
                relaxed = float(max(0.01, np.percentile(density, 10)))
                threshold.SetLowerThreshold(relaxed)
                threshold.SetUpperThreshold(upper)
                threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
                threshold.Update()
                threshold_output = threshold.GetOutput()

            mapper.SetInputData(threshold_output)
            mapper.SetScalarRange(0, 1)
            mapper.SetInputData(threshold_output)
            mapper.SetScalarRange(0, 1)
        else:
            # For FEA/ShapeOpt meshes without density - use surface filter
            surface = vtk.vtkDataSetSurfaceFilter()
            surface.SetInputData(grid)
            surface.Update()
            mapper.SetInputData(surface.GetOutput())
            
            # Set up coloring for stress/displacement
            dataset = surface.GetOutput()
            
            # First try to use the active scalars if set
            if dataset.GetPointData().GetScalars() is not None:
                scalars = dataset.GetPointData().GetScalars()
                min_val, max_val = scalars.GetRange()
                
                # Avoid degenerate range
                if max_val - min_val < 1e-10:
                    max_val = min_val + 1.0
                    
                mapper.SetScalarRange(min_val, max_val)
                
                lut = vtk.vtkLookupTable()
                lut.SetHueRange(0.667, 0.0)  # Blue to Red
                lut.Build()
                mapper.SetLookupTable(lut)
                
                scalar_name = scalars.GetName()
                if scalar_name == "VonMises":
                    self._update_scalar_bar("Von Mises Stress (MPa)", min_val, max_val, lut)
                elif scalar_name == "Displacement":
                    self._update_scalar_bar("Displacement (mm)", min_val, max_val, lut)
            
            # If no active scalars but we have stress data, set it up manually
            elif stress is not None and len(stress) > 0:
                min_s, max_s = float(np.min(stress)), float(np.max(stress))
                
                # Avoid degenerate range
                if max_s - min_s < 1e-10:
                    max_s = min_s + 1.0
                
                mapper.SetScalarRange(min_s, max_s)
                mapper.SetScalarModeToUsePointData()
                
                lut = vtk.vtkLookupTable()
                lut.SetHueRange(0.667, 0.0)
                lut.Build()
                mapper.SetLookupTable(lut)
                
                self._update_scalar_bar("Von Mises Stress (MPa)", min_s, max_s, lut)
            
            else:
                # Plain mesh - use default color (handled by actor properties below)
                pass

        # If we have density (TopOpt), we might still want to color by Stress!
        if density is not None and stress is not None and visualization_mode == 'Von Mises Stress':
            # Configure mapper to use point data scalars (VonMises) instead of cell data (Density)
            # The threshold filter should have passed point arrays through (if SetPassPointArrays was called)
            
            # Get the current mapper input and try to set active scalars
            mapper_input = mapper.GetInput()
            if mapper_input is not None and mapper_input.GetPointData() is not None:
                # Set VonMises as the active scalar array for coloring
                mapper_input.GetPointData().SetActiveScalars("VonMises")

            
            # Tell mapper to use point data for coloring, not cell data
            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray("VonMises")
            
            # Avoid degenerate range
            min_s, max_s = float(np.min(stress)), float(np.max(stress))
            if max_s - min_s < 1e-10:
                max_s = min_s + 1.0
            
            mapper.SetScalarRange(min_s, max_s)
            
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)  # Blue to Red
            lut.Build()
            mapper.SetLookupTable(lut)
            self._update_scalar_bar("Von Mises Stress (MPa)", min_s, max_s, lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Styling
        if density is not None and visualization_mode == 'Density':
            actor.GetProperty().SetColor(0.9, 0.7, 0.1) # Gold for optimized shape
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()
        elif stress is not None or displacement is not None:
            # For FEA results, show colored surface with mesh boundaries
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()
        else:
            actor.GetProperty().SetColor(0.8, 0.4, 0.4) # Reddish for plain meshes
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()
        
        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)
        
        # 4. Debug Visualization (Loads/Constraints)
        if isinstance(data, dict):
            if 'debug_loads' in data and data['debug_loads']:
                for load in data['debug_loads']:
                    # Scale logic: try to be relative to mesh size?
                    # For now just pass raw values
                    self._add_arrow(load['start'], load['vector'], color=(1, 1, 0), scale=1.0)
            
            if 'debug_constraints' in data and data['debug_constraints']:
                for const in data['debug_constraints']:
                    self._add_cube_marker(const['pos'], color=(1, 0, 0), size=2.0)
            
            # 5. Recovered Shape (smooth marching-cubes surface)
            recovered = data.get('recovered_shape')
            if recovered is not None and isinstance(recovered, dict):
                verts_r = recovered.get('vertices')
                faces_r = recovered.get('faces')
                if verts_r is not None and faces_r is not None and len(verts_r) > 0 and len(faces_r) > 0:
                    # Build VTK polydata from marching-cubes surface
                    rec_points = vtk.vtkPoints()
                    for v in verts_r:
                        rec_points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
                    
                    rec_polys = vtk.vtkCellArray()
                    for f in faces_r:
                        rec_polys.InsertNextCell(3)
                        rec_polys.InsertCellPoint(int(f[0]))
                        rec_polys.InsertCellPoint(int(f[1]))
                        rec_polys.InsertCellPoint(int(f[2]))
                    
                    rec_pd = vtk.vtkPolyData()
                    rec_pd.SetPoints(rec_points)
                    rec_pd.SetPolys(rec_polys)
                    
                    # Smooth normals for a polished look
                    rec_normals = vtk.vtkPolyDataNormals()
                    rec_normals.SetInputData(rec_pd)
                    rec_normals.SetFeatureAngle(60.0)
                    rec_normals.Update()
                    
                    rec_mapper = vtk.vtkPolyDataMapper()
                    rec_mapper.SetInputConnection(rec_normals.GetOutputPort())
                    
                    rec_actor = vtk.vtkActor()
                    rec_actor.SetMapper(rec_mapper)
                    rec_actor.GetProperty().SetColor(0.85, 0.85, 0.90)   # Light silver
                    rec_actor.GetProperty().SetSpecular(0.6)
                    rec_actor.GetProperty().SetSpecularPower(30)
                    rec_actor.GetProperty().SetOpacity(0.85)
                    rec_actor.GetProperty().SetRepresentationToSurface()
                    rec_actor.GetProperty().EdgeVisibilityOff()
                    
                    self.renderer.AddActor(rec_actor)
                    self.actors.append(rec_actor)
                    
                    # Hide the raw density mesh when recovered shape is available
                    if self.current_actor is not None:
                        self.current_actor.VisibilityOff()

        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()