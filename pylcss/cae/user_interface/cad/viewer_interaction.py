# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Camera and pointer behaviour that matches standard engineering CAD conventions.

Three differences account for most of how an interactive viewer *feels* in professional
CAD and FEA tools, none of them visual:

* the wheel zooms toward the cursor, not the centre of the window;
* standard views ease into place instead of snapping;
* the entity under the pointer highlights before it is clicked.

Everything here is display-only.
"""

from __future__ import annotations

import logging

import numpy as np
import vtk
from PySide6 import QtCore

__all__ = ["CadInteractorStyle", "ViewerInteractionMixin"]

_LOG = logging.getLogger(__name__)

_ZOOM_STEP = 1.12
_ANIM_MS = 16
_ANIM_STEPS = 16
_HOVER_INTERVAL_MS = 45


def _normalize(vector):
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def _camera_frame(direction, view_up):
    """Build an orthonormal right/up/back camera frame."""
    back = _normalize(np.asarray(direction, dtype=float))
    up = np.asarray(view_up, dtype=float)
    up = _normalize(up - back * float(np.dot(up, back)))
    if float(np.linalg.norm(up)) < 1e-12:
        fallback = np.array((0.0, 0.0, 1.0))
        if abs(float(np.dot(back, fallback))) > 0.95:
            fallback = np.array((0.0, 1.0, 0.0))
        up = _normalize(fallback - back * float(np.dot(fallback, back)))
    right = _normalize(np.cross(up, back))
    up = _normalize(np.cross(back, right))
    return np.asarray((right, up, back))


def _quaternion_from_matrix(matrix):
    """Convert a 3x3 rotation matrix to a normalized (w, x, y, z) quaternion."""
    matrix = np.asarray(matrix, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        root = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                0.25 * root,
                (matrix[2, 1] - matrix[1, 2]) / root,
                (matrix[0, 2] - matrix[2, 0]) / root,
                (matrix[1, 0] - matrix[0, 1]) / root,
            )
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            root = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                (
                    (matrix[2, 1] - matrix[1, 2]) / root,
                    0.25 * root,
                    (matrix[0, 1] + matrix[1, 0]) / root,
                    (matrix[0, 2] + matrix[2, 0]) / root,
                )
            )
        elif axis == 1:
            root = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 2] - matrix[2, 0]) / root,
                    (matrix[0, 1] + matrix[1, 0]) / root,
                    0.25 * root,
                    (matrix[1, 2] + matrix[2, 1]) / root,
                )
            )
        else:
            root = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                (
                    (matrix[1, 0] - matrix[0, 1]) / root,
                    (matrix[0, 2] + matrix[2, 0]) / root,
                    (matrix[1, 2] + matrix[2, 1]) / root,
                    0.25 * root,
                )
            )
    return _normalize(quaternion)


def _matrix_from_quaternion(quaternion):
    w, x, y, z = _normalize(np.asarray(quaternion, dtype=float))
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def _slerp_quaternion(start, end, t):
    """Shortest-path spherical interpolation between unit quaternions."""
    start = _normalize(np.asarray(start, dtype=float))
    end = _normalize(np.asarray(end, dtype=float))
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize(start + (end - start) * t)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    return start * (np.sin((1.0 - t) * angle) / sine) + end * (
        np.sin(t * angle) / sine
    )


class CadInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera whose wheel zooms toward the pointer.

    VTK dollies along the view axis, so zooming in on a detail at the edge of
    the window pushes it off-screen and forces a pan.  Anchoring the zoom to
    the point under the cursor is what every CAD package does.
    """

    def __init__(self, renderer):
        super().__init__()
        self._renderer = renderer
        self.AddObserver("MouseWheelForwardEvent", self._on_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self._on_wheel_backward)

    def _on_wheel_forward(self, *_args):
        self._zoom_to_cursor(_ZOOM_STEP)

    def _on_wheel_backward(self, *_args):
        self._zoom_to_cursor(1.0 / _ZOOM_STEP)

    def _world_under_cursor(self, renderer, x, y):
        """World point under the pointer — on geometry if hit, else focal plane."""
        picker = vtk.vtkPropPicker()
        try:
            if picker.Pick(x, y, 0, renderer) and picker.GetViewProp() is not None:
                return np.asarray(picker.GetPickPosition(), dtype=float)
        except Exception:
            _LOG.debug("Prop pick failed during zoom.", exc_info=True)
        # Nothing under the pointer: unproject onto the plane through the
        # focal point so zooming over empty space still behaves predictably.
        try:
            camera = renderer.GetActiveCamera()
            renderer.SetWorldPoint(*camera.GetFocalPoint(), 1.0)
            renderer.WorldToDisplay()
            depth = renderer.GetDisplayPoint()[2]
            renderer.SetDisplayPoint(float(x), float(y), depth)
            renderer.DisplayToWorld()
            world = np.asarray(renderer.GetWorldPoint(), dtype=float)
            if abs(world[3]) > 1e-12:
                return world[:3] / world[3]
        except Exception:
            _LOG.debug("Could not unproject the cursor.", exc_info=True)
        return None

    def _zoom_to_cursor(self, factor):
        renderer = self._renderer() if callable(self._renderer) else self._renderer
        if renderer is None:
            return
        interactor = self.GetInteractor()
        if interactor is None:
            return
        camera = renderer.GetActiveCamera()
        x, y = interactor.GetEventPosition()
        target = self._world_under_cursor(renderer, x, y)

        position = np.asarray(camera.GetPosition(), dtype=float)
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        blend = 1.0 - 1.0 / factor

        try:
            if target is not None:
                camera.SetPosition(*(position + (target - position) * blend))
                camera.SetFocalPoint(*(focal + (target - focal) * blend))
            else:
                camera.Dolly(factor)
            renderer.ResetCameraClippingRange()
            interactor.Render()
        except Exception:
            _LOG.debug("Zoom-to-cursor failed.", exc_info=True)


class ViewerInteractionMixin:
    """Animated standard views and hover preselection."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _init_interaction(self):
        self._camera_anim_timer = QtCore.QTimer(self)
        self._camera_anim_timer.setInterval(_ANIM_MS)
        self._camera_anim_timer.timeout.connect(self._on_camera_anim_step)
        self._camera_anim_state = None
        self._animate_views = True

        self._hover_enabled = False
        self._hover_index = None
        self._hover_kind = None
        self._hover_actor = None
        self._hover_pending = None
        self._hover_observer_id = None
        self._hover_timer = QtCore.QTimer(self)
        self._hover_timer.setInterval(_HOVER_INTERVAL_MS)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._resolve_hover)

    def _install_cad_interactor_style(self):
        """Swap VTK's trackball for the zoom-to-cursor variant."""
        try:
            style = CadInteractorStyle(self.renderer)
            self.interactor.SetInteractorStyle(style)
            self._default_interactor_style = style
        except Exception:
            _LOG.debug("Could not install the CAD interactor style.", exc_info=True)

    # ------------------------------------------------------------------
    # animated standard views
    # ------------------------------------------------------------------
    def _animate_camera_to(self, position, focal_point, view_up, distance):
        """Ease the camera onto a standard view.

        A hard cut loses the viewer's sense of which way the part turned; the
        short interpolation is what makes a nav-cube click readable.
        """
        camera = self.renderer.GetActiveCamera()
        start_focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        start_direction = _normalize(
            np.asarray(camera.GetPosition(), dtype=float) - start_focal
        )
        start_distance = float(camera.GetDistance())
        start_up = _normalize(np.asarray(camera.GetViewUp(), dtype=float))

        end_focal = np.asarray(focal_point, dtype=float)
        end_direction = _normalize(np.asarray(position, dtype=float))
        end_up = _normalize(np.asarray(view_up, dtype=float))

        start_rotation = _quaternion_from_matrix(
            _camera_frame(start_direction, start_up)
        )
        end_rotation = _quaternion_from_matrix(_camera_frame(end_direction, end_up))

        if not self._animate_views:
            end_frame = _matrix_from_quaternion(end_rotation)
            self._apply_camera_pose(end_focal, end_frame[2], end_frame[1], distance)
            return

        self._camera_anim_state = {
            "step": 0,
            "start": (start_focal, start_rotation, start_distance),
            "end": (end_focal, end_rotation, float(distance)),
        }
        self._camera_anim_timer.start()

    def _apply_camera_pose(self, focal, direction, view_up, distance):
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*focal)
        camera.SetPosition(*(focal + direction * distance))
        camera.SetViewUp(*view_up)
        camera.OrthogonalizeViewUp()
        camera.ParallelProjectionOff()
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()
        if hasattr(self, "_nav_cube"):
            try:
                self._nav_cube.update_rotation(camera)
            except Exception:
                _LOG.debug("Optional UI operation failed.", exc_info=True)

    def _on_camera_anim_step(self):
        state = self._camera_anim_state
        if not state:
            self._camera_anim_timer.stop()
            return
        state["step"] += 1
        raw = min(1.0, state["step"] / float(_ANIM_STEPS))
        # Cosine ease-in-out — constant-velocity interpolation reads as
        # mechanical at this duration.
        t = 0.5 - 0.5 * np.cos(np.pi * raw)

        start_focal, start_rotation, start_distance = state["start"]
        end_focal, end_rotation, end_distance = state["end"]

        focal = start_focal + (end_focal - start_focal) * t
        rotation = _slerp_quaternion(start_rotation, end_rotation, t)
        frame = _matrix_from_quaternion(rotation)
        direction = frame[2]
        view_up = frame[1]
        distance = start_distance + (end_distance - start_distance) * t
        self._apply_camera_pose(focal, direction, view_up, distance)

        if raw >= 1.0:
            self._camera_anim_timer.stop()
            self._camera_anim_state = None
            # Settling the camera restores full render quality.
            if hasattr(self, "_on_interaction_end"):
                self._on_interaction_end()

    # ------------------------------------------------------------------
    # hover preselection
    # ------------------------------------------------------------------
    def _enable_hover_preselect(self, kind):
        """Start highlighting the entity under the pointer.

        Only active while a picking mode is running: outside selection the
        highlight is noise, and the per-move pick is wasted work.
        """
        self._hover_enabled = True
        self._hover_kind = kind
        if self._hover_observer_id is None:
            try:
                self._hover_observer_id = self.interactor.AddObserver(
                    "MouseMoveEvent", self._on_hover_move
                )
            except Exception:
                _LOG.debug("Could not observe pointer motion.", exc_info=True)

    def _disable_hover_preselect(self):
        self._hover_enabled = False
        self._hover_kind = None
        self._hover_timer.stop()
        if self._hover_observer_id is not None:
            try:
                self.interactor.RemoveObserver(self._hover_observer_id)
            except Exception:
                _LOG.debug("Optional UI operation failed.", exc_info=True)
            self._hover_observer_id = None
        self._clear_hover_actor()
        try:
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            _LOG.debug("Optional UI operation failed.", exc_info=True)

    def _on_hover_move(self, *_args):
        """Coalesce pointer motion — picking on every move event is too costly."""
        if not self._hover_enabled:
            return
        try:
            self._hover_pending = self.interactor.GetEventPosition()
        except Exception:
            return
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _resolve_hover(self):
        if not self._hover_enabled or self._hover_pending is None:
            return
        x, y = self._hover_pending
        index = None
        if self._hover_kind == "face" and self.current_actor is not None:
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.001)
            picker.AddPickList(self.current_actor)
            picker.PickFromListOn()
            picker.Pick(x, y, 0, self.renderer)
            cell_id = picker.GetCellId()
            if cell_id >= 0:
                index = self._face_map.get(cell_id)
        elif self._hover_kind == "edge" and self._edge_actor is not None:
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.02)
            picker.AddPickList(self._edge_actor)
            picker.PickFromListOn()
            picker.Pick(x, y, 0, self.renderer)
            cell_id = picker.GetCellId()
            if cell_id >= 0:
                index = self._edge_cell_map.get(cell_id)

        if index == self._hover_index:
            return
        self._hover_index = index
        self._clear_hover_actor()
        if index is not None:
            self._build_hover_actor(index)
        try:
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            _LOG.debug("Optional UI operation failed.", exc_info=True)

    def _build_hover_actor(self, index):
        """Draw the preselect highlight, kept distinct from the selection colour."""
        try:
            if self._hover_kind == "face":
                if index >= len(self._face_polydata_list):
                    return
                polydata = self._face_polydata_list[index]
                if polydata is None:
                    return
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                mapper.ScalarVisibilityOff()
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-3.0, -3.0)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                # Cyan-white preselect against the orange committed selection,
                # so hovering an already-selected face still reads.
                actor.GetProperty().SetColor(0.55, 0.85, 1.0)
                actor.GetProperty().SetOpacity(0.55)
                actor.GetProperty().LightingOff()
            else:
                if index >= len(self._edge_pd_list):
                    return
                polydata = self._edge_pd_list[index]
                if polydata is None:
                    return
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.55, 0.85, 1.0)
                actor.GetProperty().SetLineWidth(5.0)
                actor.GetProperty().LightingOff()
            actor.PickableOff()
            self.renderer.AddActor(actor)
            self._hover_actor = actor
        except Exception:
            _LOG.debug("Could not build the hover highlight.", exc_info=True)

    def _clear_hover_actor(self):
        if self._hover_actor is not None:
            try:
                self.renderer.RemoveActor(self._hover_actor)
            except Exception:
                _LOG.debug("Optional UI operation failed.", exc_info=True)
            self._hover_actor = None
