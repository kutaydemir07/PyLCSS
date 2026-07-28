# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""PickingMixin implementation for the CAD viewer."""

from __future__ import annotations

import logging

import numpy as np
import vtk
from PySide6 import QtCore, QtGui, QtWidgets

from .viewer_constants import (
    MESH_COMPONENT_INDEX_BASE as _MESH_COMPONENT_INDEX_BASE,
)

__all__ = ["PickingMixin"]


class PickingMixin:
    def _apply_rotation_lock(self, locked):
        """Swap the VTK interactor style so the camera can't rotate when locked.

        Qt event filters on QVTKRenderWindowInteractor aren't reliable on Windows
        (events reach VTK through native paths that bypass the filter), so we
        disable rotation at the VTK level instead by replacing the trackball
        style with a no-op style.
        """
        if locked:
            self.interactor.SetInteractorStyle(self._noop_interactor_style)
        else:
            self.interactor.SetInteractorStyle(self._default_interactor_style)

    def _on_vtk_pick(self, obj, event):
        """VTK LeftButtonPressEvent observer — dispatches to face/edge picker."""
        if not self._rotation_locked:
            return  # in Rotate mode, clicks orbit the camera instead of picking
        x, y = self.interactor.GetEventPosition()
        mods = QtWidgets.QApplication.keyboardModifiers()
        ctrl = bool(mods & QtCore.Qt.ControlModifier)
        if self._vertex_picking_mode:
            self._do_vertex_pick(x, y, ctrl)
        elif self._edge_picking_mode:
            self._do_edge_pick(x, y, ctrl)
        elif self._picking_mode:
            self._do_face_pick(x, y, ctrl)

    def _on_lock_toggled(self, locked):
        """Toggle between locked-pick and free-rotate modes."""
        self._rotation_locked = locked
        self._apply_rotation_lock(locked)
        if self._btn_lock is not None:
            self._btn_lock.setText("🔒 Lock" if locked else "🔓 Rotate")

    def enable_picking_mode(self, multi_select=True):
        """Switch the viewer into face-selection picking mode."""
        self._picking_mode = True
        self._multi_select = multi_select
        self._picked_face_indices = []
        self._picked_occ_faces = []
        self._clear_highlight_actors()

        # Show toolbar, reset lock to locked state
        self._picking_toolbar.show()
        self._pick_count_lbl.setText("0 selected")
        if self._btn_pick_faces is not None:
            self._btn_pick_faces.setEnabled(False)
        if self._btn_lock is not None:
            self._btn_lock.setChecked(True)

        # Lock rotation at the VTK level (swap to no-op interactor style) and
        # listen for left-button presses to drive the picker.
        self._rotation_locked = True
        self._apply_rotation_lock(True)
        if self._pick_callback_id is None:
            self._pick_callback_id = self.interactor.AddObserver(
                "LeftButtonPressEvent", self._on_vtk_pick
            )

        self.vtkWidget.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

    def disable_picking_mode(self):
        """Exit picking mode, restore free orbit."""
        self._picking_mode = False
        self._picking_toolbar.hide()
        self._clear_highlight_actors()

        self._rotation_locked = False
        self._apply_rotation_lock(False)
        if self._pick_callback_id is not None:
            try:
                self.interactor.RemoveObserver(self._pick_callback_id)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            self._pick_callback_id = None

        self.vtkWidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        if self._btn_pick_faces is not None:
            self._btn_pick_faces.setEnabled(True)

    def _do_face_pick(self, vtk_x, vtk_y, ctrl_held):
        """Run the face cell-picker at the given VTK viewport coordinates."""
        if not self._picking_mode:
            return

        x, y = vtk_x, vtk_y

        # Use cell picker for face selection
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.001)
        # Restrict hits to the rendered engineering surface. Without a pick
        # list, a BC overlay/highlight actor can return its own local cell id,
        # which may accidentally resolve to a different geometry face.
        if self.current_actor is not None:
            picker.AddPickList(self.current_actor)
            picker.PickFromListOn()
        picker.Pick(x, y, 0, self.renderer)

        cell_id = picker.GetCellId()
        if cell_id < 0:
            return  # Missed — no geometry at this pixel

        face_idx = self._face_map.get(cell_id, None)
        if face_idx is None:
            return

        if not ctrl_held:
            # Replace selection
            self._picked_face_indices = [face_idx]
            self._picked_occ_faces = (
                [self._all_occ_faces[face_idx]]
                if face_idx < len(self._all_occ_faces)
                else []
            )
        else:
            # Toggle face in/out of selection
            if face_idx in self._picked_face_indices:
                self._picked_face_indices.remove(face_idx)
                if face_idx < len(self._all_occ_faces):
                    face = self._all_occ_faces[face_idx]
                    if face in self._picked_occ_faces:
                        self._picked_occ_faces.remove(face)
            else:
                self._picked_face_indices.append(face_idx)
                if face_idx < len(self._all_occ_faces):
                    self._picked_occ_faces.append(self._all_occ_faces[face_idx])

        # Update highlights
        self._update_highlight_actors()
        n = len(self._picked_face_indices)
        self._pick_count_lbl.setText(f"{n} face{'s' if n != 1 else ''} selected")

    def _update_highlight_actors(self):
        """Re-render face highlights for currently selected faces."""
        self._clear_highlight_actors()

        for face_idx in self._picked_face_indices:
            if face_idx >= len(self._face_polydata_list):
                continue
            face_pd = self._face_polydata_list[face_idx]
            if face_pd is None:
                continue

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(face_pd)
            mapper.ScalarVisibilityOff()
            # Pull the highlight slightly toward the camera so it always wins
            # the depth test against the coincident base-face triangles — without
            # this, the overlay flickers / vanishes during camera rotation.
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.55, 0.0)  # Orange highlight
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetLineWidth(2.0)
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(1.0, 0.85, 0.2)

            self.renderer.AddActor(actor)
            self._highlight_actors.append(actor)

        self.vtkWidget.GetRenderWindow().Render()

    def _clear_highlight_actors(self):
        """Remove all highlight actors from the scene."""
        for actor in self._highlight_actors:
            self.renderer.RemoveActor(actor)
        self._highlight_actors = []

    def enable_edge_picking_mode(self, multi_select=True):
        """Switch the viewer into edge-selection picking mode."""
        self._edge_picking_mode = True
        self._multi_select = multi_select
        self._picked_edge_indices = []
        self._picked_occ_edges = []
        self._clear_edge_highlight_actors()

        # Show picking toolbar with edge label
        self._pick_icon_lbl.setText("Edge Picking Mode  --  Click edges to select")
        self._picking_toolbar.show()
        self._pick_count_lbl.setText("0 selected")

        # Make edge actor pickable and visible so the picker can hit it
        if self._edge_actor is not None:
            self._edge_actor.SetPickable(1)
            self._edge_actor.SetVisibility(1)

        # Lock rotation via interactor-style swap; same observer used for faces
        # dispatches edge picks because _edge_picking_mode is True.
        if self._btn_lock is not None:
            self._btn_lock.setChecked(True)
        self._rotation_locked = True
        self._apply_rotation_lock(True)
        if self._edge_pick_callback_id is None:
            self._edge_pick_callback_id = self.interactor.AddObserver(
                "LeftButtonPressEvent", self._on_vtk_pick
            )
        self.vtkWidget.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

    def disable_edge_picking_mode(self):
        """Exit edge picking mode and restore free orbit."""
        self._edge_picking_mode = False
        self._picking_toolbar.hide()
        self._clear_edge_highlight_actors()
        self._pick_icon_lbl.setText("Face Picking Mode  --  Click faces to select")

        self._rotation_locked = False
        self._apply_rotation_lock(False)
        if self._edge_pick_callback_id is not None:
            try:
                self.interactor.RemoveObserver(self._edge_pick_callback_id)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            self._edge_pick_callback_id = None

        if self._edge_actor is not None:
            self._edge_actor.SetPickable(0)
            self._edge_actor.SetVisibility(1 if self._show_edges else 0)

        self.vtkWidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    def _do_edge_pick(self, vtk_x, vtk_y, ctrl_held):
        """Run the edge cell-picker at the given VTK viewport coordinates."""
        if not self._edge_picking_mode:
            return

        if self._edge_actor is None:
            return

        x, y = vtk_x, vtk_y

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.02)  # wider tolerance helps hit thin lines
        picker.AddPickList(self._edge_actor)
        picker.PickFromListOn()
        picker.Pick(x, y, 0, self.renderer)

        cell_id = picker.GetCellId()
        if cell_id < 0:
            return

        edge_idx = self._edge_cell_map.get(cell_id, None)
        if edge_idx is None:
            return

        if not ctrl_held:
            self._picked_edge_indices = [edge_idx]
            self._picked_occ_edges = (
                [self._all_occ_edges[edge_idx]]
                if edge_idx < len(self._all_occ_edges)
                else []
            )
        else:
            if edge_idx in self._picked_edge_indices:
                self._picked_edge_indices.remove(edge_idx)
                if edge_idx < len(self._all_occ_edges):
                    e = self._all_occ_edges[edge_idx]
                    if e in self._picked_occ_edges:
                        self._picked_occ_edges.remove(e)
            else:
                self._picked_edge_indices.append(edge_idx)
                if edge_idx < len(self._all_occ_edges):
                    self._picked_occ_edges.append(self._all_occ_edges[edge_idx])

        self._update_edge_highlight_actors()
        n = len(self._picked_edge_indices)
        self._pick_count_lbl.setText(f"{n} edge{'s' if n != 1 else ''} selected")

    def _update_edge_highlight_actors(self):
        """Re-render edge highlights for currently selected edges."""
        self._clear_edge_highlight_actors()
        for edge_idx in self._picked_edge_indices:
            if edge_idx >= len(self._edge_pd_list):
                continue
            edge_pd = self._edge_pd_list[edge_idx]
            if edge_pd is None:
                continue
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(edge_pd)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.9, 1.0)  # cyan
            actor.GetProperty().SetLineWidth(4.0)
            actor.GetProperty().LightingOff()
            self.renderer.AddActor(actor)
            self._edge_highlight_actors.append(actor)
        self.vtkWidget.GetRenderWindow().Render()

    def _clear_edge_highlight_actors(self):
        """Remove all edge highlight actors from the scene."""
        for actor in self._edge_highlight_actors:
            self.renderer.RemoveActor(actor)
        self._edge_highlight_actors = []

    def enable_vertex_picking_mode(self, multi_select=True):
        """Switch the viewer into CAD-vertex picking mode."""
        if self._vertex_actor is None or not self._all_occ_vertices:
            return False
        self._vertex_picking_mode = True
        self._multi_select = multi_select
        self._picked_vertex_indices = []
        self._picked_occ_vertices = []
        self._clear_vertex_highlight_actors()
        self._pick_icon_lbl.setText(
            "Vertex Picking Mode  --  Click corner points to select"
        )
        self._picking_toolbar.show()
        self._pick_count_lbl.setText("0 selected")
        self._vertex_actor.SetVisibility(1)
        self._vertex_actor.SetPickable(1)
        if self._btn_lock is not None:
            self._btn_lock.setChecked(True)
        self._rotation_locked = True
        self._apply_rotation_lock(True)
        if self._vertex_pick_callback_id is None:
            self._vertex_pick_callback_id = self.interactor.AddObserver(
                "LeftButtonPressEvent", self._on_vtk_pick
            )
        self.vtkWidget.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.vtkWidget.GetRenderWindow().Render()
        return True

    def disable_vertex_picking_mode(self):
        """Exit vertex picking mode and restore free orbit."""
        self._vertex_picking_mode = False
        self._picking_toolbar.hide()
        self._clear_vertex_highlight_actors()
        self._pick_icon_lbl.setText("Face Picking Mode  --  Click faces to select")
        self._rotation_locked = False
        self._apply_rotation_lock(False)
        if self._vertex_pick_callback_id is not None:
            try:
                self.interactor.RemoveObserver(self._vertex_pick_callback_id)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            self._vertex_pick_callback_id = None
        if self._vertex_actor is not None:
            self._vertex_actor.SetPickable(0)
            self._vertex_actor.SetVisibility(0)
        self.vtkWidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.vtkWidget.GetRenderWindow().Render()

    def _do_vertex_pick(self, vtk_x, vtk_y, ctrl_held):
        """Pick one OCC vertex from the dedicated point actor."""
        if not self._vertex_picking_mode or self._vertex_actor is None:
            return
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.025)
        picker.AddPickList(self._vertex_actor)
        picker.PickFromListOn()
        picker.Pick(vtk_x, vtk_y, 0, self.renderer)
        vertex_idx = int(picker.GetPointId())
        if not 0 <= vertex_idx < len(self._all_occ_vertices):
            return
        if not ctrl_held:
            self._picked_vertex_indices = [vertex_idx]
        elif vertex_idx in self._picked_vertex_indices:
            self._picked_vertex_indices.remove(vertex_idx)
        else:
            self._picked_vertex_indices.append(vertex_idx)
        self._picked_occ_vertices = [
            self._all_occ_vertices[index] for index in self._picked_vertex_indices
        ]
        self._update_vertex_highlight_actors()
        count = len(self._picked_vertex_indices)
        self._pick_count_lbl.setText(
            f"{count} vertex{'es' if count != 1 else ''} selected"
        )

    def _update_vertex_highlight_actors(self):
        self._clear_vertex_highlight_actors()
        for index in self._picked_vertex_indices:
            if not 0 <= index < len(self._all_occ_vertices):
                continue
            try:
                center = self._all_occ_vertices[index].Center()
                point = (float(center.x), float(center.y), float(center.z))
            except Exception:
                continue
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*point)
            bounds = self.renderer.ComputeVisiblePropBounds()
            diagonal = (
                float(
                    np.linalg.norm(
                        [
                            bounds[1] - bounds[0],
                            bounds[3] - bounds[2],
                            bounds[5] - bounds[4],
                        ]
                    )
                )
                if bounds and len(bounds) == 6
                else 1.0
            )
            sphere.SetRadius(max(0.001, 0.012 * diagonal))
            sphere.SetThetaResolution(16)
            sphere.SetPhiResolution(12)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.55, 0.0)
            actor.GetProperty().SetAmbient(1.0)
            actor.GetProperty().SetDiffuse(0.0)
            self.renderer.AddActor(actor)
            self._vertex_highlight_actors.append(actor)
        self.vtkWidget.GetRenderWindow().Render()

    def _clear_vertex_highlight_actors(self):
        for actor in self._vertex_highlight_actors:
            self.renderer.RemoveActor(actor)
        self._vertex_highlight_actors = []

    def _on_pick_done(self):
        """User confirmed picking — emit signal and exit picking mode."""
        if self._vertex_picking_mode:
            picked = list(self._picked_occ_vertices)
            self.disable_vertex_picking_mode()
            self.vertex_picked.emit(picked)
            return
        if self._edge_picking_mode:
            picked = list(self._picked_occ_edges)
            self.disable_edge_picking_mode()
            self.edge_picked.emit(picked)
            return
        picked = list(self._picked_occ_faces)
        picked_indices = list(self._picked_face_indices)
        self.disable_picking_mode()
        # Re-add highlights in a passive/dimmed color to keep them visible
        for idx in picked_indices:
            if (
                0 <= idx < len(self._face_polydata_list)
                and self._face_polydata_list[idx] is not None
            ):
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(self._face_polydata_list[idx])
                mapper.ScalarVisibilityOff()
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.6, 0.1)
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().EdgeVisibilityOn()
                actor.GetProperty().SetEdgeColor(1.0, 0.85, 0.2)
                actor.GetProperty().SetLineWidth(1.5)
                self.renderer.AddActor(actor)
                self.actors.append(actor)
        self.vtkWidget.GetRenderWindow().Render()
        self.face_picked.emit(picked)

    def _on_pick_cancel(self):
        """User cancelled picking."""
        if self._vertex_picking_mode:
            self.disable_vertex_picking_mode()
        elif self._edge_picking_mode:
            self.disable_edge_picking_mode()
        else:
            self.disable_picking_mode()
        self.picking_cancelled.emit()

    def highlight_faces(self, face_indices):
        """Public method to highlight specific face indices matching current geometry."""
        self._clear_highlight_actors()
        for stored_idx in face_indices:
            idx = int(stored_idx)
            if idx >= _MESH_COMPONENT_INDEX_BASE:
                idx = next(
                    (
                        face_idx
                        for face_idx, face in enumerate(self._all_occ_faces)
                        if isinstance(face, dict)
                        and int(face.get("stored_index", -1)) == int(stored_idx)
                    ),
                    -1,
                )
            if (
                0 <= idx < len(self._face_polydata_list)
                and self._face_polydata_list[idx] is not None
            ):
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(self._face_polydata_list[idx])
                mapper.ScalarVisibilityOff()
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.6, 0.1)  # Dim orange
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().EdgeVisibilityOn()
                actor.GetProperty().SetEdgeColor(1.0, 0.85, 0.2)
                actor.GetProperty().SetLineWidth(1.5)
                self.renderer.AddActor(actor)
                self.actors.append(actor)
        self.vtkWidget.GetRenderWindow().Render()

    def highlight_edges(self, edge_indices):
        """Highlight stored CAD-edge indices on the current shape."""
        self._picked_edge_indices = [
            int(index)
            for index in edge_indices
            if 0 <= int(index) < len(self._all_occ_edges)
        ]
        self._update_edge_highlight_actors()

    def highlight_vertices(self, vertex_indices):
        """Highlight stored CAD-vertex indices on the current shape."""
        self._picked_vertex_indices = [
            int(index)
            for index in vertex_indices
            if 0 <= int(index) < len(self._all_occ_vertices)
        ]
        self._update_vertex_highlight_actors()

    def confirm_picking(self):
        self._on_pick_done()

    def cancel_picking(self):
        self._on_pick_cancel()
