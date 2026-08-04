# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""BoundaryOverlayMixin implementation for the CAD viewer."""

from __future__ import annotations

import logging

import numpy as np
import vtk

from .boundary_visuals import (
    BC_PALETTE,
    compact_constraint_label,
    constraint_color,
    farthest_spatial_samples,
    legend_caption,
    merge_magnitude_range,
    overlay_scale,
)

__all__ = ["BoundaryOverlayMixin"]


def _deduplicate_gravity_vectors(load_vectors):
    """Collapse identical component gravity glyphs into one global overlay."""
    merged = []
    gravity_groups = {}
    for raw_entry in load_vectors or []:
        if not (
            isinstance(raw_entry, dict)
            and str(raw_entry.get("quantity_type") or "").lower() == "gravity"
        ):
            merged.append(raw_entry)
            continue
        try:
            vector = np.asarray(raw_entry.get("vector"), dtype=float).reshape(3)
            magnitude = float(
                raw_entry.get("magnitude_N", np.linalg.norm(vector))
            )
            vector_norm = float(np.linalg.norm(vector))
            center = np.asarray(raw_entry.get("centroid"), dtype=float).reshape(3)
        except (TypeError, ValueError):
            merged.append(raw_entry)
            continue
        if vector_norm <= 1.0e-12 or not np.all(np.isfinite(center)):
            merged.append(raw_entry)
            continue
        direction = vector / vector_norm
        key = (
            tuple(float(value) for value in np.round(direction, 8)),
            round(magnitude, 8),
        )
        existing = gravity_groups.get(key)
        if existing is None:
            entry = dict(raw_entry)
            entry["_gravity_overlay_count"] = 1
            gravity_groups[key] = entry
            merged.append(entry)
            continue
        count = int(existing.get("_gravity_overlay_count") or 1)
        previous_center = np.asarray(existing["centroid"], dtype=float)
        existing["centroid"] = (
            (count * previous_center + center) / (count + 1)
        ).tolist()
        existing["_gravity_overlay_count"] = count + 1
    return merged


def _aggregate_colocated_load_case_vectors(load_vectors):
    """Show one force envelope at a location shared by several load cases."""
    passthrough = []
    grouped = {}
    for raw_entry in load_vectors or []:
        if not (
            isinstance(raw_entry, dict)
            and raw_entry.get("load_case")
            and str(raw_entry.get("quantity_type") or "force").lower() == "force"
        ):
            passthrough.append(raw_entry)
            continue
        try:
            centroid = np.asarray(raw_entry.get("centroid"), dtype=float).reshape(3)
            magnitude = float(
                raw_entry.get(
                    "magnitude_N",
                    np.linalg.norm(np.asarray(raw_entry.get("vector"), dtype=float)),
                )
            )
        except (TypeError, ValueError):
            passthrough.append(raw_entry)
            continue
        key = tuple(float(value) for value in np.round(centroid, decimals=5))
        try:
            vector = np.asarray(raw_entry.get("vector"), dtype=float).reshape(3)
            vector_norm = float(np.linalg.norm(vector))
            direction_key = (
                tuple(float(value) for value in np.round(vector / vector_norm, 3))
                if vector_norm > 1.0e-12
                else (0.0, 0.0, 0.0)
            )
        except (TypeError, ValueError):
            direction_key = (0.0, 0.0, 0.0)
        grouped.setdefault(key, []).append((raw_entry, magnitude, direction_key))

    merged = list(passthrough)
    for entries in grouped.values():
        case_names = {
            str(entry.get("load_case") or "")
            for entry, _magnitude, _direction in entries
            if entry.get("load_case")
        }
        if len(case_names) <= 1:
            merged.extend(entry for entry, _magnitude, _direction in entries)
            continue

        direction_groups = {}
        for entry, magnitude, direction_key in entries:
            direction_groups.setdefault(direction_key, []).append((entry, magnitude))
        representatives = [
            max(direction_entries, key=lambda item: item[1])
            for direction_entries in direction_groups.values()
        ]
        representative, _ = max(representatives, key=lambda item: item[1])
        envelope = dict(representative)
        magnitudes = [
            magnitude for _entry, magnitude, _direction in entries
        ]
        envelope["_load_case_count"] = len(case_names)
        envelope["_force_envelope_min_N"] = float(min(magnitudes))
        envelope["_force_envelope_max_N"] = float(max(magnitudes))
        envelope["label"] = None
        merged.append(envelope)
        for direction_entry, _magnitude in representatives:
            if direction_entry is representative:
                continue
            secondary = dict(direction_entry)
            secondary["label"] = None
            secondary["_suppress_label"] = True
            merged.append(secondary)
    return merged


class BoundaryOverlayMixin:
    def render_bc_overlays(
        self, constraint_faces=None, load_faces=None, load_vectors=None
    ):
        """
        Render semantic engineering conditions over CAD, mesh, or results.

        Surface/volume regions are depth-correct. Their outlines, DOF stops,
        load arrows, wall/joint graphics, labels, and the HUD remain visible in
        the foreground. The three legacy list arguments remain API-compatible,
        while dictionary entries carry exact condition meaning and metadata.
        """
        # ── Remove all previous BC overlay actors ──────────────────────────────
        for actor in list(self._bc_overlay_actors):
            self.renderer.RemoveActor(actor)
            self.bc_renderer.RemoveActor(actor)
            try:
                self.renderer.RemoveActor2D(actor)
                self.bc_renderer.RemoveActor2D(actor)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            # Also pull out of the generic actors list if it ended up there
            if actor in self.actors:
                self.actors.remove(actor)
        self._bc_overlay_actors = []
        # A result may have been rendered before exact graph conditions were
        # collected. Remove its approximate fallback markers before drawing
        # the exact overlays, otherwise two glyphs/labels occupy the same point.
        for actor in list(self.actors):
            if not getattr(actor, "_bc_result_fallback", False):
                continue
            self.renderer.RemoveActor(actor)
            self.bc_renderer.RemoveActor(actor)
            self.actors.remove(actor)
        load_vectors = _deduplicate_gravity_vectors(load_vectors)
        load_vectors = _aggregate_colocated_load_case_vectors(load_vectors)

        def _track_actor(actor, *, foreground=True, actor_2d=False):
            """Register one BC prop in the depth-correct or always-visible pass."""
            if actor is None:
                return None
            actor._bc_overlay = True
            target = self.bc_renderer if foreground else self.renderer
            if actor_2d:
                target.AddActor2D(actor)
            else:
                target.AddActor(actor)
            self._bc_overlay_actors.append(actor)
            return actor

        def _annotation_color(color):
            """Keep semantic hues legible against the active label background."""
            rgb = np.clip(np.asarray(color, dtype=float), 0.0, 1.0)
            luminance = float(np.dot(rgb, (0.2126, 0.7152, 0.0722)))
            if self._light_mode and luminance > 0.42:
                rgb *= 0.56
            elif not self._light_mode and luminance < 0.42:
                rgb = 0.72 * rgb + 0.28
            return tuple(float(component) for component in rgb)

        def _triangulate_face(occ_face):
            try:
                tri = occ_face.tessellate(tolerance=0.15, angularTolerance=0.3)
                if isinstance(tri, dict):
                    verts = tri.get("vertices") or tri.get("verts") or []
                    tris = tri.get("triangles") or tri.get("faces") or []
                else:
                    verts, tris = tri[0], tri[1]
                if not verts:
                    return None, None
                np_verts = np.array([[v.x, v.y, v.z] for v in verts], dtype=float)
                return np_verts, tris
            except Exception:
                return None, None

        def _face_to_polydata(occ_face):
            verts, tris = _triangulate_face(occ_face)
            if verts is None or tris is None:
                return None
            try:
                pts = vtk.vtkPoints()
                polys = vtk.vtkCellArray()
                for v in verts:
                    pts.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
                for t in tris:
                    polys.InsertNextCell(3)
                    polys.InsertCellPoint(t[0])
                    polys.InsertCellPoint(t[1])
                    polys.InsertCellPoint(t[2])
                pd = vtk.vtkPolyData()
                pd.SetPoints(pts)
                pd.SetPolys(polys)
                return pd
            except Exception:
                return None

        def _mesh_selection_to_polydata(selection):
            if not isinstance(selection, dict):
                return None
            try:
                verts = np.asarray(selection.get("surface_vertices") or [], dtype=float)
                tris = np.asarray(selection.get("surface_triangles") or [], dtype=int)
                if (
                    verts.ndim == 2
                    and verts.shape[1] >= 3
                    and len(verts) > 0
                    and tris.ndim == 2
                    and tris.shape[1] >= 3
                    and len(tris) > 0
                ):
                    pts = vtk.vtkPoints()
                    polys = vtk.vtkCellArray()
                    for v in verts[:, :3]:
                        pts.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
                    for tri in tris[:, :3]:
                        if np.any(tri < 0) or np.any(tri >= len(verts)):
                            continue
                        polys.InsertNextCell(3)
                        polys.InsertCellPoint(int(tri[0]))
                        polys.InsertCellPoint(int(tri[1]))
                        polys.InsertCellPoint(int(tri[2]))
                    if polys.GetNumberOfCells() > 0:
                        pd = vtk.vtkPolyData()
                        pd.SetPoints(pts)
                        pd.SetPolys(polys)
                        return pd

                # Open shell ends have no cap triangles. Draw their true free
                # edges rather than collapsing the selection to one centroid.
                edge_verts = np.asarray(
                    selection.get("edge_vertices") or [], dtype=float
                )
                edge_lines = np.asarray(
                    selection.get("edge_lines") or [], dtype=int
                )
                if (
                    edge_verts.ndim == 2
                    and edge_verts.shape[1] >= 3
                    and len(edge_verts) > 0
                    and edge_lines.ndim == 2
                    and edge_lines.shape[1] >= 2
                    and len(edge_lines) > 0
                ):
                    pts = vtk.vtkPoints()
                    lines = vtk.vtkCellArray()
                    for vertex in edge_verts[:, :3]:
                        pts.InsertNextPoint(
                            float(vertex[0]),
                            float(vertex[1]),
                            float(vertex[2]),
                        )
                    for line in edge_lines[:, :2]:
                        if np.any(line < 0) or np.any(line >= len(edge_verts)):
                            continue
                        lines.InsertNextCell(2)
                        lines.InsertCellPoint(int(line[0]))
                        lines.InsertCellPoint(int(line[1]))
                    if lines.GetNumberOfCells() > 0:
                        pd = vtk.vtkPolyData()
                        pd.SetPoints(pts)
                        pd.SetLines(lines)
                        return pd

                selected = np.asarray(
                    selection.get("selected_vertices") or [], dtype=float
                )
                if (
                    selected.ndim == 2
                    and selected.shape[1] >= 3
                    and len(selected) > 0
                ):
                    pts = vtk.vtkPoints()
                    vertices = vtk.vtkCellArray()
                    for index, vertex in enumerate(selected[:, :3]):
                        pts.InsertNextPoint(
                            float(vertex[0]),
                            float(vertex[1]),
                            float(vertex[2]),
                        )
                        vertices.InsertNextCell(1)
                        vertices.InsertCellPoint(index)
                    pd = vtk.vtkPolyData()
                    pd.SetPoints(pts)
                    pd.SetVerts(vertices)
                    return pd
                return None
            except Exception:
                return None

        def _mesh_selection_sample_points(selection, max_points=6):
            if not isinstance(selection, dict):
                return []
            try:
                verts = np.asarray(selection.get("surface_vertices") or [], dtype=float)
                tris = np.asarray(selection.get("surface_triangles") or [], dtype=int)
                if (
                    verts.ndim != 2
                    or verts.shape[1] < 3
                    or tris.ndim != 2
                    or tris.shape[1] < 3
                ):
                    edge_vertices = np.asarray(
                        selection.get("edge_vertices")
                        or selection.get("selected_vertices")
                        or [],
                        dtype=float,
                    )
                    if (
                        edge_vertices.ndim == 2
                        and edge_vertices.shape[1] >= 3
                        and len(edge_vertices) > 0
                    ):
                        return [
                            point.tolist()
                            for point in farthest_spatial_samples(
                                edge_vertices[:, :3],
                                max_points=max_points,
                                min_spacing=0.65 * _overlay_scale(),
                            )
                        ]
                    return selection.get("points") or []
                centroids = []
                areas = []
                for tri in tris[:, :3]:
                    if np.any(tri < 0) or np.any(tri >= len(verts)):
                        continue
                    pa, pb, pc = (
                        verts[int(tri[0])],
                        verts[int(tri[1])],
                        verts[int(tri[2])],
                    )
                    centroids.append((pa + pb + pc) / 3.0)
                    areas.append(0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa)))
                if not centroids:
                    return selection.get("points") or []
                bbox = selection.get("bbox") or {}
                try:
                    dx = float(bbox["xmax"]) - float(bbox["xmin"])
                    dy = float(bbox["ymax"]) - float(bbox["ymin"])
                    dz = float(bbox["zmax"]) - float(bbox["zmin"])
                    min_spacing = 0.12 * ((dx * dx + dy * dy + dz * dz) ** 0.5)
                except Exception:
                    min_spacing = 0.0
                return [
                    point.tolist()
                    for point in farthest_spatial_samples(
                        centroids,
                        weights=areas,
                        max_points=max_points,
                        min_spacing=min_spacing,
                    )
                ]
            except Exception:
                return selection.get("points") or []

        def _face_boundary_polydata(occ_face):
            verts, tris = _triangulate_face(occ_face)
            if verts is None or tris is None:
                return None
            try:
                edge_counts = {}
                for tri in tris:
                    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                    for i0, i1 in ((a, b), (b, c), (c, a)):
                        edge = (min(i0, i1), max(i0, i1))
                        edge_counts[edge] = edge_counts.get(edge, 0) + 1
                boundary_edges = [
                    edge for edge, count in edge_counts.items() if count == 1
                ]
                if not boundary_edges:
                    return None
                pts = vtk.vtkPoints()
                lines = vtk.vtkCellArray()
                for vert in verts:
                    pts.InsertNextPoint(float(vert[0]), float(vert[1]), float(vert[2]))
                for i0, i1 in boundary_edges:
                    lines.InsertNextCell(2)
                    lines.InsertCellPoint(i0)
                    lines.InsertCellPoint(i1)
                pd = vtk.vtkPolyData()
                pd.SetPoints(pts)
                pd.SetLines(lines)
                return pd
            except Exception:
                return None

        def _sample_face_points(occ_face, max_points=6):
            verts, tris = _triangulate_face(occ_face)
            if verts is None or tris is None or len(tris) == 0:
                return []
            try:
                tri_centroids = []
                tri_areas = []
                for tri in tris:
                    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                    pa, pb, pc = verts[a], verts[b], verts[c]
                    tri_centroids.append((pa + pb + pc) / 3.0)
                    tri_areas.append(0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa)))
                if not tri_centroids:
                    return []
                try:
                    bb = occ_face.BoundingBox()
                    diag = (
                        (bb.xmax - bb.xmin) ** 2
                        + (bb.ymax - bb.ymin) ** 2
                        + (bb.zmax - bb.zmin) ** 2
                    ) ** 0.5
                    min_spacing = 0.12 * diag
                except Exception:
                    min_spacing = 0.0
                return [
                    point.tolist()
                    for point in farthest_spatial_samples(
                        tri_centroids,
                        weights=tri_areas,
                        max_points=max_points,
                        min_spacing=min_spacing,
                    )
                ]
            except Exception:
                return []

        def _shape_type(entity):
            try:
                return str(entity.ShapeType())
            except Exception:
                return "Face"

        def _sample_edge_points(edge, max_points=6):
            points = []
            try:
                from OCP.BRepAdaptor import BRepAdaptor_Curve
                from OCP.GCPnts import GCPnts_UniformAbscissa

                curve = BRepAdaptor_Curve(edge.wrapped)
                sampler = GCPnts_UniformAbscissa(curve, max(2, int(max_points)))
                for index in range(1, sampler.NbPoints() + 1):
                    point = curve.Value(sampler.Parameter(index))
                    points.append([point.X(), point.Y(), point.Z()])
            except Exception:
                try:
                    start, end = edge.startPoint(), edge.endPoint()
                    points = [
                        [start.x, start.y, start.z],
                        [end.x, end.y, end.z],
                    ]
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            return points

        def _sample_entity_points(entity, max_points=6):
            if isinstance(entity, dict):
                return entity.get("points") or (
                    [entity.get("center")] if entity.get("center") is not None else []
                )
            kind = _shape_type(entity)
            if kind == "Edge":
                return _sample_edge_points(entity, max_points=max_points)
            if kind == "Vertex":
                try:
                    center = entity.Center()
                    return [[center.x, center.y, center.z]]
                except Exception:
                    return []
            return _sample_face_points(entity, max_points=max_points)

        def _add_entity_overlay(entity, color, line_width=3.0, opacity=1.0):
            """Draw a face patch, CAD edge, or vertex without changing meaning."""
            kind = _shape_type(entity)
            if kind == "Edge":
                sampled = _sample_edge_points(entity, max_points=24)
                if len(sampled) < 2:
                    return
                points = vtk.vtkPoints()
                lines = vtk.vtkCellArray()
                for point in sampled:
                    points.InsertNextPoint(*point)
                for index in range(len(sampled) - 1):
                    lines.InsertNextCell(2)
                    lines.InsertCellPoint(index)
                    lines.InsertCellPoint(index + 1)
                poly_data = vtk.vtkPolyData()
                poly_data.SetPoints(points)
                poly_data.SetLines(lines)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly_data)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetLineWidth(max(4.0, line_width))
                actor.GetProperty().LightingOff()
            elif kind == "Vertex":
                points = _sample_entity_points(entity, max_points=1)
                if not points:
                    return
                source = vtk.vtkSphereSource()
                source.SetCenter(*points[0])
                source.SetRadius(0.22 * _overlay_scale())
                source.SetThetaResolution(20)
                source.SetPhiResolution(14)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(source.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetAmbient(1.0)
                actor.GetProperty().SetDiffuse(0.0)
            elif kind == "Face":
                _add_face_outline(
                    entity,
                    color=color,
                    line_width=line_width,
                    opacity=opacity,
                )
                return
            else:
                # Closed non-design solids and compounds have no boundary edge
                # in their tessellation. Render a transparent, depth-correct
                # volume plus always-visible feature/silhouette edges.
                pd = _face_to_polydata(entity)
                if pd is None:
                    return
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(pd)
                mapper.ScalarVisibilityOff()
                fill_actor = vtk.vtkActor()
                fill_actor.SetMapper(mapper)
                fill_actor.GetProperty().SetColor(*color)
                fill_actor.GetProperty().SetOpacity(max(0.16, min(opacity, 0.32)))
                fill_actor.GetProperty().LightingOff()
                _track_actor(fill_actor, foreground=False)

                features = vtk.vtkFeatureEdges()
                features.SetInputData(pd)
                features.BoundaryEdgesOn()
                features.FeatureEdgesOn()
                features.NonManifoldEdgesOn()
                features.ManifoldEdgesOff()
                features.SetFeatureAngle(32.0)
                edge_mapper = vtk.vtkPolyDataMapper()
                edge_mapper.SetInputConnection(features.GetOutputPort())
                edge_mapper.ScalarVisibilityOff()
                edge_actor = vtk.vtkActor()
                edge_actor.SetMapper(edge_mapper)
                edge_actor.GetProperty().SetColor(*color)
                edge_actor.GetProperty().SetLineWidth(max(3.0, line_width))
                edge_actor.GetProperty().LightingOff()
                _track_actor(edge_actor, foreground=True)
                return
            _track_actor(actor, foreground=True)

        def _pressure_samples(entity, max_points=5):
            """Return outward-normal samples for a CAD or mesh surface."""
            if isinstance(entity, dict):
                verts = np.asarray(entity.get("surface_vertices") or [], dtype=float)
                tris = np.asarray(entity.get("surface_triangles") or [], dtype=int)
            else:
                verts, tris = _triangulate_face(entity)
                if verts is None:
                    return []
                tris = np.asarray(tris, dtype=int)
            if verts.ndim != 2 or tris.ndim != 2 or len(verts) == 0 or len(tris) == 0:
                return []
            try:
                bounds = self.current_actor.GetBounds()
                body_center = np.asarray(
                    [
                        0.5 * (bounds[0] + bounds[1]),
                        0.5 * (bounds[2] + bounds[3]),
                        0.5 * (bounds[4] + bounds[5]),
                    ]
                )
            except Exception:
                body_center = np.mean(verts[:, :3], axis=0)
            candidates = []
            for tri in tris[:, :3]:
                if np.any(tri < 0) or np.any(tri >= len(verts)):
                    continue
                a, b, c = verts[tri[0], :3], verts[tri[1], :3], verts[tri[2], :3]
                normal = np.cross(b - a, c - a)
                area_twice = float(np.linalg.norm(normal))
                if area_twice <= 1e-12:
                    continue
                normal /= area_twice
                center = (a + b + c) / 3.0
                if float(np.dot(normal, center - body_center)) < 0.0:
                    normal *= -1.0
                candidates.append((area_twice, center, normal))
            if not candidates:
                return []
            centers = [candidate[1] for candidate in candidates]
            areas = [candidate[0] for candidate in candidates]
            sampled_centers = farthest_spatial_samples(
                centers,
                weights=areas,
                max_points=max_points,
                min_spacing=0.65 * _overlay_scale(),
            )
            selected = []
            for sampled in sampled_centers:
                nearest = min(
                    candidates,
                    key=lambda candidate: float(np.linalg.norm(candidate[1] - sampled)),
                )
                selected.append((nearest[1], nearest[2]))
            return selected

        def _add_bc_label(position, text, color):
            if not text:
                return
            try:
                actor = vtk.vtkBillboardTextActor3D()
                actor.SetPosition(*position)
                actor.SetInput(str(text))
                actor.SetPickable(0)
                actor._bc_text_label = True
                actor.SetVisibility(1 if self._bc_text_visible() else 0)
                prop = actor.GetTextProperty()
                prop.SetFontSize(14)
                prop.BoldOn()
                prop.SetColor(*_annotation_color(color))
                prop.SetJustificationToCentered()
                prop.SetVerticalJustificationToBottom()
                # A backed annotation stays legible over stress contours, CAD
                # bodies, and either application theme.
                try:
                    prop.SetBackgroundColor(
                        *(0.95, 0.97, 0.99) if self._light_mode else (0.04, 0.05, 0.07)
                    )
                    prop.SetBackgroundOpacity(0.88)
                except AttributeError:
                    pass
                _track_actor(actor, foreground=True)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )

        def _selection_label_position(selection, fallback):
            """Move a region label just outside the displayed body."""
            try:
                position = np.asarray(
                    selection.get("centroid")
                    or selection.get("center")
                    or fallback,
                    dtype=float,
                )[:3]
                bounds = self.current_actor.GetBounds()
                body_center = np.asarray(
                    [
                        0.5 * (bounds[0] + bounds[1]),
                        0.5 * (bounds[2] + bounds[3]),
                        0.5 * (bounds[4] + bounds[5]),
                    ],
                    dtype=float,
                )
                outward = position - body_center
                length = float(np.linalg.norm(outward))
                if length <= 1e-9:
                    outward = np.asarray((0.0, 0.0, 1.0))
                else:
                    outward /= length
                return position + 0.55 * _overlay_scale() * outward
            except Exception:
                return np.asarray(fallback, dtype=float)

        def _plane_axes(normal):
            normal = np.asarray(normal, dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            helper = (
                np.asarray((0.0, 0.0, 1.0))
                if abs(float(normal[2])) < 0.9
                else np.asarray((0.0, 1.0, 0.0))
            )
            axis_u = np.cross(normal, helper)
            axis_u /= max(float(np.linalg.norm(axis_u)), 1e-12)
            axis_v = np.cross(normal, axis_u)
            return axis_u, axis_v

        def _add_face_outline(occ_face, color, line_width=2.5, opacity=1.0):
            boundary_pd = _face_boundary_polydata(occ_face)
            if boundary_pd is not None:
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(boundary_pd)
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetLineWidth(line_width)
                actor.GetProperty().SetOpacity(opacity)
                actor.GetProperty().LightingOff()
                _track_actor(actor, foreground=True)
                return

            pd = _face_to_polydata(occ_face)
            if pd is None:
                return
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetOpacity(0.18)
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(*color)
            actor.GetProperty().SetLineWidth(line_width)
            _track_actor(actor, foreground=True)

        def _add_face_overlay(occ_face, color, opacity=0.12):
            pd = _face_to_polydata(occ_face)
            if pd is None:
                return
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetOpacity(opacity)
            actor.GetProperty().EdgeVisibilityOff()
            actor.GetProperty().LightingOff()
            _track_actor(actor, foreground=False)

        def _add_mesh_selection_overlay(
            selection, color, opacity=0.16, line_width=2.5, edge_color=None
        ):
            pd = _mesh_selection_to_polydata(selection)
            if pd is None:
                return
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)
            mapper.ScalarVisibilityOff()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            has_surface = pd.GetNumberOfPolys() > 0
            actor.GetProperty().SetOpacity(opacity if has_surface else 1.0)
            actor.GetProperty().EdgeVisibilityOff()
            actor.GetProperty().LightingOff()
            if not has_surface:
                actor.GetProperty().SetLineWidth(line_width)
                actor.GetProperty().SetPointSize(max(4.0, 1.5 * line_width))
            _track_actor(actor, foreground=not has_surface)

            if not has_surface:
                return

            edge_filter = vtk.vtkFeatureEdges()
            edge_filter.SetInputData(pd)
            edge_filter.BoundaryEdgesOn()
            edge_filter.FeatureEdgesOn()
            edge_filter.NonManifoldEdgesOn()
            edge_filter.ManifoldEdgesOff()
            edge_filter.SetFeatureAngle(25.0)
            edge_mapper = vtk.vtkPolyDataMapper()
            edge_mapper.SetInputConnection(edge_filter.GetOutputPort())
            edge_mapper.ScalarVisibilityOff()
            edge_actor = vtk.vtkActor()
            edge_actor.SetMapper(edge_mapper)
            edge_actor.GetProperty().SetColor(*(edge_color or color))
            edge_actor.GetProperty().SetLineWidth(line_width)
            edge_actor.GetProperty().LightingOff()
            _track_actor(edge_actor, foreground=True)

        def _add_bc_arrow(start, vector, color):
            """Arrow tagged as a BC overlay so it is cleaned up with the rest.

            The arrow is drawn with length == np.linalg.norm(vector), pointing
            in the direction of vector, with its BASE at *start*.  The caller
            is responsible for choosing *start* so the tip lands where desired.
            """
            import numpy as _np

            length = _np.linalg.norm(vector)
            if length < 1e-9:
                return
            arrow_source = vtk.vtkArrowSource()
            arrow_source.SetShaftRadius(0.032)
            arrow_source.SetTipRadius(0.105)
            arrow_source.SetTipLength(0.28)
            arrow_source.SetShaftResolution(20)
            arrow_source.SetTipResolution(24)
            norm_vec = vector / length
            transform = vtk.vtkTransform()
            transform.Translate(start[0], start[1], start[2])
            default_dir = _np.array([1.0, 0.0, 0.0])
            axis = _np.cross(default_dir, norm_vec)
            angle_deg = _np.degrees(
                _np.arccos(_np.clip(_np.dot(default_dir, norm_vec), -1.0, 1.0))
            )
            if _np.linalg.norm(axis) > 1e-6:
                transform.RotateWXYZ(angle_deg, axis)
            elif _np.dot(default_dir, norm_vec) < 0:
                transform.RotateWXYZ(180, [0, 1, 0])
            # Scale so the rendered arrow length == the magnitude of *vector*.
            transform.Scale(length, length, length)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(arrow_source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetUserTransform(transform)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetAmbient(0.75)
            actor.GetProperty().SetDiffuse(0.25)
            _track_actor(actor, foreground=True)

        def _overlay_scale():
            try:
                actor = self.current_actor
                bounds = actor.GetBounds() if actor is not None else None
                return overlay_scale(bounds)
            except Exception:
                return 0.035

        def _add_constraint_direction_overlay(occ_face, viz_meta, color):
            """Draw only prescribed motion as an arrow.

            Zero-valued restraints use T-stop glyphs. Using arrowheads for
            restraints made them indistinguishable from applied loads.
            """
            sample_points = _sample_entity_points(occ_face, max_points=3)
            if not sample_points:
                return

            arrow_len = _overlay_scale()
            disp_vals = viz_meta.get("displacement")

            if disp_vals is not None:
                disp_vec = np.array(disp_vals, dtype=float)
                if np.linalg.norm(disp_vec) > 1e-12:
                    vis_vec = disp_vec / np.linalg.norm(disp_vec) * arrow_len
                    for point in sample_points:
                        start = np.array(point, dtype=float)
                        _add_bc_arrow(start.tolist(), vis_vec.tolist(), color)

        def _constraint_label(viz_meta):
            return compact_constraint_label(viz_meta)

        def _add_polyline(points_in, color, line_width=3.0):
            clean_points = [
                np.asarray(point, dtype=float)
                for point in points_in
                if point is not None
            ]
            if len(clean_points) < 2:
                return
            vtk_points = vtk.vtkPoints()
            vtk_lines = vtk.vtkCellArray()
            for point in clean_points:
                vtk_points.InsertNextPoint(*point)
            for index in range(len(clean_points) - 1):
                vtk_lines.InsertNextCell(2)
                vtk_lines.InsertCellPoint(index)
                vtk_lines.InsertCellPoint(index + 1)
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(vtk_points)
            poly_data.SetLines(vtk_lines)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(line_width)
            actor.GetProperty().LightingOff()
            _track_actor(actor, foreground=True)

        def _add_anchor_marker(position, color, radius):
            source = vtk.vtkSphereSource()
            source.SetCenter(*position)
            source.SetRadius(radius)
            source.SetThetaResolution(20)
            source.SetPhiResolution(14)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetAmbient(1.0)
            actor.GetProperty().SetDiffuse(0.0)
            _track_actor(actor, foreground=True)

        def _add_joint_link(viz_meta):
            anchors = [
                np.asarray(point, dtype=float)
                for point in (viz_meta.get("anchors") or [])
                if point is not None
            ]
            if len(anchors) < 2:
                return
            color = BC_PALETTE["joint"]
            _add_polyline(anchors[:2], color, line_width=5.0)
            for anchor in anchors[:2]:
                _add_anchor_marker(anchor, color, 0.22 * _overlay_scale())
            midpoint = 0.5 * (anchors[0] + anchors[1])
            joint_type = str(viz_meta.get("joint_type") or "Joint").upper()
            axis_name = str(viz_meta.get("axis") or "X").strip().upper()
            if joint_type in ("REVOLUTE", "PRISMATIC"):
                axis_vector = {
                    "X": np.asarray((1.0, 0.0, 0.0)),
                    "Y": np.asarray((0.0, 1.0, 0.0)),
                    "Z": np.asarray((0.0, 0.0, 1.0)),
                }.get(axis_name, np.asarray((1.0, 0.0, 0.0)))
                half_axis = 0.9 * _overlay_scale() * axis_vector
                _add_bc_arrow(
                    (midpoint - half_axis).tolist(),
                    (2.0 * half_axis).tolist(),
                    color,
                )
            label = joint_type
            if joint_type in ("REVOLUTE", "PRISMATIC"):
                label += f" · {axis_name}"
            _add_bc_label(midpoint, label, color)

        def _add_impact_wall(item):
            if getattr(self, "_crash_wall_info", None):
                return
            velocity = np.asarray(item.get("vector") or (0.0, 0.0, 0.0), dtype=float)
            magnitude = float(np.linalg.norm(velocity))
            if magnitude <= 1e-12:
                return
            normal = velocity / magnitude
            try:
                bounds = self.current_actor.GetBounds()
                body_center = np.asarray(
                    [
                        0.5 * (bounds[0] + bounds[1]),
                        0.5 * (bounds[2] + bounds[3]),
                        0.5 * (bounds[4] + bounds[5]),
                    ]
                )
                half_extents = np.asarray(
                    [
                        0.5 * (bounds[1] - bounds[0]),
                        0.5 * (bounds[3] - bounds[2]),
                        0.5 * (bounds[5] - bounds[4]),
                    ]
                )
            except Exception:
                body_center = np.asarray(item.get("centroid") or (0.0, 0.0, 0.0))
                half_extents = np.ones(3) * _overlay_scale()

            scope = str(item.get("application_scope") or "").strip().lower()
            gap = abs(float(item.get("wall_gap_mm") or 0.0))
            if gap <= 1e-12:
                gap = 0.35 * _overlay_scale()
            if scope.startswith("moving body"):
                projection_radius = float(np.dot(np.abs(normal), half_extents))
                center = body_center + normal * (projection_radius + gap)
                wall_label = "FIXED WALL"
            else:
                center = np.asarray(item.get("centroid") or body_center)
                center = center - normal * gap
                wall_label = (
                    "MOVING WALL" if scope.startswith("prescribed") else "IMPACTOR"
                )

            axis_u, axis_v = _plane_axes(normal)
            half_u = float(np.dot(np.abs(axis_u), half_extents))
            half_v = float(np.dot(np.abs(axis_v), half_extents))
            entity = item.get("face")
            bbox = entity.get("bbox") if isinstance(entity, dict) else None
            if isinstance(bbox, dict):
                try:
                    bbox_corners = np.asarray(
                        [
                            (x, y, z)
                            for x in (float(bbox["xmin"]), float(bbox["xmax"]))
                            for y in (float(bbox["ymin"]), float(bbox["ymax"]))
                            for z in (float(bbox["zmin"]), float(bbox["zmax"]))
                        ],
                        dtype=float,
                    )
                    half_u = 0.5 * float(np.ptp(bbox_corners @ axis_u))
                    half_v = 0.5 * float(np.ptp(bbox_corners @ axis_v))
                except Exception:
                    pass
            minimum_half = 0.75 * _overlay_scale()
            half_u = max(minimum_half, 1.08 * half_u)
            half_v = max(minimum_half, 1.08 * half_v)
            corner_0 = center - axis_u * half_u - axis_v * half_v
            corner_1 = center + axis_u * half_u - axis_v * half_v
            corner_2 = center - axis_u * half_u + axis_v * half_v

            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(*corner_0)
            plane.SetPoint1(*corner_1)
            plane.SetPoint2(*corner_2)
            plane.SetXResolution(4)
            plane.SetYResolution(4)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*BC_PALETTE["impact"])
            actor.GetProperty().SetOpacity(0.22)
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(*BC_PALETTE["impact"])
            actor.GetProperty().SetLineWidth(2.5)
            actor.GetProperty().LightingOff()
            _track_actor(actor, foreground=False)

            corners = [
                corner_0,
                corner_1,
                center + axis_u * half_u + axis_v * half_v,
                corner_2,
                corner_0,
            ]
            _add_polyline(corners, BC_PALETTE["impact"], line_width=2.5)
            wall_label_position = (
                center + 0.78 * axis_u * half_u + 0.92 * axis_v * half_v
            )
            _add_bc_label(
                wall_label_position,
                wall_label,
                BC_PALETTE["impact"],
            )

        if constraint_faces:
            for item in constraint_faces:
                if item is None:
                    continue
                if isinstance(item, dict) and item.get("pos") is not None:
                    viz_meta = item.get("viz", {}) or {}
                    if viz_meta.get("visual_kind") == "joint_link":
                        _add_joint_link(viz_meta)
                        continue
                    r, g, b = constraint_color(viz_meta)
                    mesh_selection = item.get("mesh_selection")
                    if isinstance(mesh_selection, dict):
                        _add_mesh_selection_overlay(
                            mesh_selection,
                            color=(r, g, b),
                            opacity=0.26,
                            line_width=4.0,
                            edge_color=(1.0, 1.0, 1.0),
                        )
                        fixed_dofs = viz_meta.get("fixed_dofs") or [0, 1, 2]
                        points = _mesh_selection_sample_points(
                            mesh_selection, max_points=3
                        )
                        if viz_meta.get("glyph", True):
                            for point in points:
                                before_count = len(self.actors)
                                self._add_constraint_glyph(
                                    point,
                                    fixed_dofs=fixed_dofs,
                                    color=(r, g, b),
                                    size=0.82 * _overlay_scale(),
                                )
                                for actor in self.actors[before_count:]:
                                    actor._bc_overlay = True
                                    self.renderer.RemoveActor(actor)
                                    self.bc_renderer.AddActor(actor)
                                    self._bc_overlay_actors.append(actor)
                        if points:
                            label = _constraint_label(viz_meta)
                            _add_bc_label(
                                _selection_label_position(
                                    mesh_selection,
                                    points[0],
                                ),
                                label,
                                (r, g, b),
                            )
                        continue
                    fixed_dofs = viz_meta.get("fixed_dofs") or [0, 1, 2]
                    points = item.get("points") or [item.get("pos")]
                    for point in points:
                        before_count = len(self.actors)
                        self._add_constraint_glyph(
                            point,
                            fixed_dofs=fixed_dofs,
                            color=(r, g, b),
                            size=0.82 * _overlay_scale(),
                        )
                        for actor in self.actors[before_count:]:
                            actor._bc_overlay = True
                            self.renderer.RemoveActor(actor)
                            self.bc_renderer.AddActor(actor)
                            self._bc_overlay_actors.append(actor)
                    if points:
                        _add_bc_label(points[0], _constraint_label(viz_meta), (r, g, b))
                    continue
                # item can be a bare OCC face (legacy) or a dict with 'face' + 'viz' keys
                if isinstance(item, dict):
                    occ_face = item.get("face")
                    viz_meta = item.get("viz", {})
                    r, g, b = constraint_color(viz_meta)
                    _label = (
                        viz_meta.get("constraint_type", "Fixed")
                        if viz_meta
                        else "Fixed"
                    )
                    _ = _label  # may be used for a 3D text label in the future
                else:
                    occ_face = item
                    r, g, b = BC_PALETTE["support"]
                if occ_face is not None:
                    _add_entity_overlay(occ_face, color=(r, g, b), line_width=3.0)
                    if _shape_type(occ_face) == "Face":
                        _add_face_overlay(
                            occ_face,
                            color=(r, g, b),
                            opacity=0.22,
                        )
                    if isinstance(item, dict):
                        _add_constraint_direction_overlay(occ_face, viz_meta, (r, g, b))
                        fixed_dofs = viz_meta.get("fixed_dofs") or [0, 1, 2]
                        points = _sample_entity_points(occ_face, max_points=3)
                        if viz_meta.get("glyph", True):
                            for point in points:
                                before_count = len(self.actors)
                                self._add_constraint_glyph(
                                    point,
                                    fixed_dofs=fixed_dofs,
                                    color=(r, g, b),
                                    size=0.82 * _overlay_scale(),
                                )
                                for actor in self.actors[before_count:]:
                                    actor._bc_overlay = True
                                    self.renderer.RemoveActor(actor)
                                    self.bc_renderer.AddActor(actor)
                                    self._bc_overlay_actors.append(actor)
                        if points:
                            _add_bc_label(
                                points[0],
                                _constraint_label(viz_meta),
                                (r, g, b),
                            )

        if load_faces:
            for item in load_faces:
                if item is None:
                    continue
                if isinstance(item, dict) and item.get("load_type") == "Selection":
                    entity = item.get("face")
                    color = BC_PALETTE["selection"]
                    if isinstance(entity, dict):
                        _add_mesh_selection_overlay(
                            entity, color=color, opacity=0.12, line_width=3.0
                        )
                    elif entity is not None:
                        _add_entity_overlay(entity, color=color, line_width=3.0)
                        if _shape_type(entity) == "Face":
                            _add_face_overlay(entity, color=color, opacity=0.08)
                    # The neutral outline already communicates selection; a
                    # billboard here duplicates inspector text and adds clutter.
                    continue
                if isinstance(item, dict) and item.get("load_type") == "Impact":
                    entity = item.get("face")
                    color = BC_PALETTE["impact"]
                    if isinstance(entity, dict):
                        _add_mesh_selection_overlay(
                            entity,
                            color=color,
                            opacity=0.20,
                            line_width=4.0,
                            edge_color=(1.0, 1.0, 1.0),
                        )
                    elif entity is not None:
                        _add_entity_overlay(entity, color=color, line_width=4.0)
                        if _shape_type(entity) == "Face":
                            _add_face_overlay(entity, color=color, opacity=0.16)
                    points = _sample_entity_points(entity, max_points=1)
                    if points:
                        impact_position = np.asarray(
                            item.get("centroid") or points[0],
                            dtype=float,
                        )
                        velocity = np.asarray(
                            item.get("vector") or (0.0, 0.0, 0.0),
                            dtype=float,
                        )
                        if float(np.linalg.norm(velocity)) > 1e-12:
                            _axis_u, axis_v = _plane_axes(velocity)
                            impact_position = (
                                impact_position
                                + 0.60 * _overlay_scale() * axis_v
                            )
                        _add_bc_label(impact_position, "IMPACT", color)
                    _add_impact_wall(item)
                    continue
                if isinstance(item, dict) and item.get("load_type") == "Heat":
                    entity = item.get("face")
                    heat_color = (1.0, 0.36, 0.12)
                    if isinstance(entity, dict):
                        _add_mesh_selection_overlay(
                            entity,
                            color=heat_color,
                            opacity=0.28,
                            line_width=3.0,
                        )
                    elif entity is not None:
                        _add_entity_overlay(
                            entity,
                            color=heat_color,
                            line_width=3.0,
                        )
                        if _shape_type(entity) == "Face":
                            _add_face_overlay(
                                entity,
                                color=heat_color,
                                opacity=0.14,
                            )
                    points = _sample_entity_points(entity, max_points=1)
                    if points:
                        _add_bc_label(
                            points[0],
                            item.get("label") or "HEAT",
                            heat_color,
                        )
                    continue
                if isinstance(item, dict) and item.get("load_type") == "Pressure":
                    entity = item.get("face")
                    pressure_color = (1.0, 0.24, 0.73)
                    if isinstance(entity, dict):
                        _add_mesh_selection_overlay(
                            entity,
                            color=pressure_color,
                            opacity=0.28,
                            line_width=3.0,
                        )
                    elif entity is not None:
                        _add_face_outline(
                            entity,
                            color=pressure_color,
                            line_width=3.0,
                        )
                        _add_face_overlay(
                            entity,
                            color=pressure_color,
                            opacity=0.14,
                        )
                    inward = str(item.get("direction") or "Inward").lower() == "inward"
                    samples = _pressure_samples(entity, max_points=5)
                    arrow_len = 0.9 * _overlay_scale()
                    for point, outward_normal in samples:
                        direction = -outward_normal if inward else outward_normal
                        start = point - direction * arrow_len if inward else point
                        _add_bc_arrow(
                            start.tolist(),
                            (direction * arrow_len).tolist(),
                            pressure_color,
                        )
                    if samples:
                        pressure = float(item.get("pressure_mpa") or 0.0)
                        _add_bc_label(
                            samples[0][0],
                            f"P {pressure:.3g} MPa",
                            pressure_color,
                        )
                    continue
                if isinstance(item, dict):
                    mesh_selection = item.get("mesh_selection")
                    if mesh_selection is None and (
                        item.get("node_ids") is not None
                        or item.get("surface_triangles") is not None
                    ):
                        mesh_selection = item
                    if isinstance(mesh_selection, dict):
                        _add_mesh_selection_overlay(
                            mesh_selection,
                            color=(1.0, 0.85, 0.0),
                            opacity=0.18,
                            line_width=2.5,
                        )
                        continue
                occ_face = item.get("face") if isinstance(item, dict) else item
                if occ_face is not None:
                    _add_entity_overlay(
                        occ_face,
                        color=(1.0, 0.85, 0.0),
                        line_width=2.5,
                    )
                    if _shape_type(occ_face) == "Face":
                        _add_face_overlay(
                            occ_face,
                            color=(1.0, 0.85, 0.0),
                            opacity=0.08,
                        )

        if load_vectors:
            import math as _math

            for entry in load_vectors:
                if isinstance(entry, dict):
                    centroid = entry["centroid"]
                    occ_face = entry.get("face")
                    vec = np.array(entry["vector"], dtype=float)
                    force_mag = float(entry.get("magnitude_N", np.linalg.norm(vec)))
                else:
                    # legacy (centroid, vec) tuple
                    centroid, vec = entry
                    occ_face = None
                    force_mag = float(np.linalg.norm(vec))
                vector_norm = float(np.linalg.norm(vec))
                if vector_norm <= 1e-12:
                    continue
                quantity_type = (
                    str(entry.get("quantity_type") or "force").lower()
                    if isinstance(entry, dict)
                    else "force"
                )
                arrow_color = BC_PALETTE.get(quantity_type, BC_PALETTE["force"])
                base_len = _overlay_scale()
                if quantity_type == "force":
                    log_s = float(
                        np.clip(_math.log10(force_mag + 1.0) / 6.0, 0.15, 1.0)
                    )
                    arrow_len = base_len * (1.8 + 1.2 * log_s)
                elif quantity_type == "gravity":
                    arrow_len = base_len * 2.8
                else:
                    arrow_len = base_len * 3.2
                unit_vec = vec / vector_norm
                vis_vec = unit_vec * arrow_len
                sample_points = (
                    _sample_entity_points(occ_face, max_points=3)
                    if occ_face is not None
                    else []
                )
                if not sample_points and isinstance(entry, dict):
                    sample_points = _mesh_selection_sample_points(
                        entry.get("mesh_selection"), max_points=3
                    )
                if not sample_points and isinstance(entry, dict):
                    sample_points = entry.get("points") or []
                if sample_points:
                    scaled_vec = unit_vec * max(
                        0.60 * arrow_len, 1.4 * _overlay_scale()
                    )
                    for point in sample_points:
                        arrow_start = np.array(point, dtype=float) - scaled_vec
                        _add_bc_arrow(
                            arrow_start.tolist(), scaled_vec, color=arrow_color
                        )
                else:
                    arrow_start = np.array(centroid, dtype=float) - vis_vec
                    _add_bc_arrow(arrow_start.tolist(), vis_vec, color=arrow_color)
                if isinstance(entry, dict):
                    if entry.get("_suppress_label"):
                        continue
                    label = entry.get("label")
                    case_count = int(entry.get("_load_case_count") or 0)
                    if case_count > 1:
                        envelope = merge_magnitude_range(
                            self._format_force_magnitude(
                                float(entry.get("_force_envelope_min_N") or 0.0)
                            ),
                            self._format_force_magnitude(
                                float(entry.get("_force_envelope_max_N") or 0.0)
                            ),
                        )
                        label = f"F {envelope} ({case_count} LC)"
                    if not label:
                        label = self._format_force_magnitude(force_mag)
                    label_position = (
                        centroid
                        if case_count > 1
                        else sample_points[0]
                        if sample_points
                        else centroid
                    )
                    if quantity_type == "velocity":
                        label_position = (
                            np.asarray(centroid, dtype=float)
                            - unit_vec * max(
                                0.85 * arrow_len,
                                2.1 * _overlay_scale(),
                            )
                        )
                    _add_bc_label(label_position, label, arrow_color)

        legend_entries = []
        if constraint_faces:
            constraint_legends = []
            for item in constraint_faces:
                viz = item.get("viz", {}) if isinstance(item, dict) else {}
                label = legend_caption(viz)
                color = constraint_color(viz)
                if (label, color) not in constraint_legends:
                    constraint_legends.append((label, color))
            legend_entries.extend(constraint_legends)
        if any(
            isinstance(item, dict) and item.get("load_type") == "Pressure"
            for item in (load_faces or [])
        ):
            legend_entries.append(("PRESSURE", BC_PALETTE["pressure"]))
        if any(
            isinstance(item, dict) and item.get("load_type") == "Heat"
            for item in (load_faces or [])
        ):
            legend_entries.append(("HEAT", BC_PALETTE["heat"]))
        if any(
            isinstance(item, dict) and item.get("load_type") == "Impact"
            for item in (load_faces or [])
        ):
            legend_entries.append(("IMPACT", BC_PALETTE["impact"]))
        if any(
            isinstance(item, dict) and item.get("load_type") == "Selection"
            for item in (load_faces or [])
        ):
            legend_entries.append(("SELECTION", BC_PALETTE["selection"]))
        if any(
            not isinstance(entry, dict)
            or entry.get("quantity_type", "force") == "force"
            for entry in (load_vectors or [])
        ):
            legend_entries.append(("FORCE", BC_PALETTE["force"]))
        if any(
            isinstance(entry, dict) and entry.get("quantity_type") == "velocity"
            for entry in (load_vectors or [])
        ):
            legend_entries.append(("VELOCITY", BC_PALETTE["velocity"]))
        if any(
            isinstance(entry, dict) and entry.get("quantity_type") == "gravity"
            for entry in (load_vectors or [])
        ):
            legend_entries.append(("GRAVITY", BC_PALETTE["gravity"]))

        if legend_entries:
            panel_background = (
                (0.95, 0.97, 0.99) if self._light_mode else (0.04, 0.05, 0.07)
            )
            title = vtk.vtkTextActor()
            title.SetInput("BOUNDARY CONDITIONS")
            title.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            title.SetPosition(0.015, 0.975)
            title.GetTextProperty().SetFontSize(15)
            title.GetTextProperty().BoldOn()
            title.GetTextProperty().SetVerticalJustificationToTop()
            title.GetTextProperty().SetColor(
                *getattr(self, "_scalar_text_color", (0.95, 0.95, 0.95))
            )
            title.GetTextProperty().SetBackgroundColor(*panel_background)
            title.GetTextProperty().SetBackgroundOpacity(0.92)
            _track_actor(title, foreground=True, actor_2d=True)

            for index, (label, color) in enumerate(legend_entries):
                actor = vtk.vtkTextActor()
                actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                actor.SetInput(label)
                actor.SetPosition(0.017, 0.922 - 0.041 * index)
                prop = actor.GetTextProperty()
                prop.SetFontSize(13)
                prop.BoldOn()
                prop.SetVerticalJustificationToTop()
                prop.SetColor(*_annotation_color(color))
                prop.SetBackgroundColor(*panel_background)
                prop.SetBackgroundOpacity(0.86)
                _track_actor(actor, foreground=True, actor_2d=True)

        # Every prop above was just created; the display switches decide what
        # of it is actually drawn.
        self._apply_bc_visibility(render=False)
        self.vtkWidget.GetRenderWindow().Render()

    def _add_arrow(self, start, vector, color=(1, 1, 0), scale=1.0):
        """Add a 3D arrow to the scene representing a vector."""
        arrow_source = vtk.vtkArrowSource()

        length = np.linalg.norm(vector)
        if length < 1e-9:
            return

        visual_length = 5.0 * scale

        normalized_vector = vector / length

        transform = vtk.vtkTransform()
        transform.Translate(start[0], start[1], start[2])

        default_dir = np.array([1.0, 0.0, 0.0])
        axis = np.cross(default_dir, normalized_vector)
        angle_rad = np.arccos(
            np.clip(np.dot(default_dir, normalized_vector), -1.0, 1.0)
        )
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
        """Add a cube marker (e.g. for constraints).  Legacy path; new code
        should use ``_add_constraint_glyph`` for the industry-standard look."""
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

    def _add_constraint_glyph(
        self, pos, fixed_dofs=None, color=(0.16, 0.47, 1.0), size=2.5
    ):
        """Draw explicit translational restraint symbols.

        Every constrained global degree of freedom is represented by a T-stop:
        a stem along that axis ending at a transverse stop bar. Unlike arrows,
        it cannot be mistaken for an applied force, and 1/2/3-DOF restraints
        remain distinct without relying on color.
        """
        fixed_dofs = sorted(
            {
                int(index)
                for index in (fixed_dofs or [])
                if isinstance(index, (int, np.integer)) and 0 <= int(index) < 3
            }
        )
        if not fixed_dofs:
            return

        origin = np.asarray(pos, dtype=float)
        axes = (
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
        )
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        def _line(point_a, point_b):
            index_a = points.InsertNextPoint(*point_a)
            index_b = points.InsertNextPoint(*point_b)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(index_a)
            lines.InsertCellPoint(index_b)

        for dof in fixed_dofs:
            direction = axes[dof]
            cross_a = axes[(dof + 1) % 3]
            cross_b = axes[(dof + 2) % 3]
            stop_center = origin - direction * (0.95 * size)
            _line(origin, stop_center)
            _line(
                stop_center - cross_a * (0.42 * size),
                stop_center + cross_a * (0.42 * size),
            )
            _line(
                stop_center - cross_b * (0.42 * size),
                stop_center + cross_b * (0.42 * size),
            )
            hatch_center = stop_center - direction * (0.18 * size)
            _line(
                hatch_center - cross_a * (0.29 * size),
                hatch_center + cross_a * (0.29 * size),
            )

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(5.0)
        actor.GetProperty().LightingOff()
        actor._bc_kind = "restraint"
        self.renderer.AddActor(actor)
        self.actors.append(actor)

        anchor = vtk.vtkSphereSource()
        anchor.SetCenter(*origin)
        anchor.SetRadius(0.14 * size)
        anchor.SetThetaResolution(18)
        anchor.SetPhiResolution(12)
        anchor_mapper = vtk.vtkPolyDataMapper()
        anchor_mapper.SetInputConnection(anchor.GetOutputPort())
        anchor_actor = vtk.vtkActor()
        anchor_actor.SetMapper(anchor_mapper)
        anchor_actor.GetProperty().SetColor(*color)
        anchor_actor.GetProperty().SetAmbient(1.0)
        anchor_actor.GetProperty().SetDiffuse(0.0)
        anchor_actor._bc_kind = "restraint_anchor"
        self.renderer.AddActor(anchor_actor)
        self.actors.append(anchor_actor)

    def _add_force_label(self, pos, magnitude_text, color=(1.0, 0.85, 0.2)):
        """Add a billboard text label at a 3D point (used for force magnitudes)."""
        try:
            actor = vtk.vtkBillboardTextActor3D()
            actor.SetPosition(pos[0], pos[1], pos[2])
            actor.SetInput(str(magnitude_text))
            actor._bc_text_label = True
            actor.SetVisibility(1 if self._bc_text_visible() else 0)
            prop = actor.GetTextProperty()
            prop.SetFontSize(14)
            prop.SetBold(True)
            prop.SetColor(color)
            prop.SetJustificationToCentered()
            prop.SetVerticalJustificationToBottom()
            prop.SetShadow(1)
            prop.SetShadowOffset(1, -1)
            self.renderer.AddActor(actor)
            self.actors.append(actor)
        except Exception:
            # Older VTK fall-back: skip the label rather than crash.
            pass

    @staticmethod
    def _format_force_magnitude(value):
        """Format Newtons compactly: 9.8e-3 N -> '9.8 mN', 12345 N -> '12.3 kN'."""
        try:
            v = abs(float(value))
        except Exception:
            return ""
        if v == 0.0:
            return "0 N"
        if v >= 1e6:
            return f"{v / 1e6:.2f} MN"
        if v >= 1e3:
            return f"{v / 1e3:.2f} kN"
        if v >= 1.0:
            return f"{v:.1f} N"
        if v >= 1e-3:
            return f"{v * 1e3:.1f} mN"
        return f"{v:.2e} N"
