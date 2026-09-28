# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.

"""Clickable camera-orientation cube for the CAD viewer."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

__all__ = ["NavCubeWidget"]


class NavCubeWidget(QtWidgets.QWidget):
    """Clickable orientation cube overlay.
    Mirrors the scene camera orientation; click any face, edge or corner
    to jump to that standard view.
    """

    view_requested = QtCore.Signal(object, object)  # (pos_tuple, up_tuple)
    roll_requested = QtCore.Signal(float)  # (angle_in_degrees)

    SIZE = 140  # widget pixel size

    # Unit cube vertices  (+X=Right, -Y=Front, +Z=Top, matching viewer axes)
    _V = np.array(
        [
            [-1, -1, -1],  # 0  FLB
            [1, -1, -1],  # 1  FRB
            [1, 1, -1],  # 2  BRB
            [-1, 1, -1],  # 3  BLB
            [-1, -1, 1],  # 4  FLT
            [1, -1, 1],  # 5  FRT
            [1, 1, 1],  # 6  BRT
            [-1, 1, 1],  # 7  BLT
        ],
        dtype=float,
    )

    # faces  (vertex_indices, label, cam_pos_norm, cam_up)
    _FACES = [
        ((4, 5, 6, 7), "+Z", (0, 0, 1), (0, 1, 0)),
        ((3, 2, 1, 0), "-Z", (0, 0, -1), (0, 1, 0)),
        ((0, 1, 5, 4), "-Y", (0, -1, 0), (0, 0, 1)),
        ((2, 3, 7, 6), "+Y", (0, 1, 0), (0, 0, 1)),
        ((1, 2, 6, 5), "+X", (1, 0, 0), (0, 0, 1)),
        ((3, 0, 4, 7), "-X", (-1, 0, 0), (0, 0, 1)),
    ]
    _FACE_N = np.array(
        [
            (0, 0, 1),
            (0, 0, -1),
            (0, -1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (-1, 0, 0),
        ],
        dtype=float,
    )

    # edges  (v0, v1, cam_pos_norm, cam_up)
    _EDGES = [
        (4, 5, (0, -1, 1), (0, 0, 1)),  # FT
        (5, 6, (1, 0, 1), (0, 0, 1)),  # RT
        (6, 7, (0, 1, 1), (0, 0, 1)),  # BT
        (7, 4, (-1, 0, 1), (0, 0, 1)),  # LT
        (0, 1, (0, -1, -1), (0, 0, 1)),  # FB
        (1, 2, (1, 0, -1), (0, 0, 1)),  # RB
        (2, 3, (0, 1, -1), (0, 0, 1)),  # BB
        (3, 0, (-1, 0, -1), (0, 0, 1)),  # LB
        (4, 0, (-1, -1, 0), (0, 0, 1)),  # FL
        (5, 1, (1, -1, 0), (0, 0, 1)),  # FR
        (6, 2, (1, 1, 0), (0, 0, 1)),  # BR
        (7, 3, (-1, 1, 0), (0, 0, 1)),  # BL
    ]

    # corners  (vertex_idx, cam_pos_norm, cam_up)
    _CORNERS = [
        (0, (-1, -1, -1), (0, 0, 1)),  # FLB
        (1, (1, -1, -1), (0, 0, 1)),  # FRB
        (2, (1, 1, -1), (0, 0, 1)),  # BRB
        (3, (-1, 1, -1), (0, 0, 1)),  # BLB
        (4, (-1, -1, 1), (0, 0, 1)),  # FLT
        (5, (1, -1, 1), (0, 0, 1)),  # FRT
        (6, (1, 1, 1), (0, 0, 1)),  # BRT
        (7, (-1, 1, 1), (0, 0, 1)),  # BLT
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self.setAutoFillBackground(False)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip("Click a face, edge, or corner to set the view orientation")
        self._rot = np.eye(3)
        self._hovered = None
        self._light_mode = False
        self._update_mask()

    def set_light_mode(self, enabled):
        self._light_mode = bool(enabled)
        self.update()

    # ── public ────────────────────────────────────────────────────────────

    def update_rotation(self, camera):
        """Synchronize the cube with a rigid, orthonormal camera frame."""
        position = np.asarray(camera.GetPosition(), dtype=float)
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        forward = self._normalized(focal - position)
        if not np.any(forward):
            return

        view_up = np.asarray(camera.GetViewUp(), dtype=float)
        view_up -= forward * float(np.dot(view_up, forward))
        view_up = self._normalized(view_up)
        right = self._normalized(np.cross(forward, view_up))
        if not np.any(right):
            return
        view_up = self._normalized(np.cross(right, forward))
        self._rot = np.asarray([right, view_up, -forward])
        self._update_mask()
        self.update()

    @staticmethod
    def _normalized(vector):
        vector = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else np.zeros(3, dtype=float)

    @classmethod
    def _view_up_for(cls, direction):
        """Return an upright, orthogonal view-up for a standard view."""
        direction = cls._normalized(direction)
        preferred = np.array((0.0, 0.0, 1.0))
        if abs(float(np.dot(direction, preferred))) > 0.95:
            preferred = np.array((0.0, 1.0, 0.0))
        view_up = preferred - direction * float(np.dot(preferred, direction))
        return tuple(float(value) for value in cls._normalized(view_up))

    # ── geometry helpers ──────────────────────────────────────────────────

    def _project(self):
        """Project 8 vertices. Returns (8,3): cols = screen_x, screen_y, depth."""
        cx = cy = self.SIZE / 2.0
        # Worst-case: camera axis aligned with body diagonal → L1-norm = sqrt(3).
        # scale × sqrt(3) must fit in (half-width − margin) for ALL orientations.
        scale = (self.SIZE / 2.0 - 8.0) / np.sqrt(3.0)
        v = self._V @ self._rot.T
        p = np.empty((8, 3))
        p[:, 0] = cx + v[:, 0] * scale
        p[:, 1] = cy - v[:, 1] * scale
        p[:, 2] = v[:, 2]
        return p

    def _update_mask(self):
        """Update the widget mask so only the cube and roll buttons are visible."""
        p = self._project()
        bitmap = QtGui.QBitmap(self.size())
        bitmap.fill(QtCore.Qt.color0)
        painter = QtGui.QPainter(bitmap)
        painter.setBrush(QtCore.Qt.color1)
        painter.setPen(QtGui.QPen(QtCore.Qt.color1, 2))
        for fi in range(6):
            vi_list = self._FACES[fi][0]
            poly = [
                QtCore.QPoint(int(round(p[v, 0])), int(round(p[v, 1])))
                for v in vi_list
            ]
            painter.drawPolygon(QtGui.QPolygon(poly))
        painter.drawRect(6, 3, 28, 28)
        painter.drawRect(self.SIZE - 34, 3, 28, 28)
        painter.end()
        self.setMask(bitmap)

    def _face_visibility(self):
        """Signed visibility of each face in the current camera frame."""
        return self._FACE_N @ self._rot[2]

    @classmethod
    def _adjacent_faces(cls, vertices):
        wanted = set(vertices)
        return [
            index
            for index, face in enumerate(cls._FACES)
            if wanted.issubset(set(face[0]))
        ]

    @staticmethod
    def _pt_in_poly(poly_xy, mx, my):
        """Point-in-polygon (works for convex polys with any winding)."""
        n = len(poly_xy)
        sign = None
        for i in range(n):
            ax, ay = poly_xy[i]
            bx, by = poly_xy[(i + 1) % n]
            cross = (bx - ax) * (my - ay) - (by - ay) * (mx - ax)
            if abs(cross) < 1e-9:
                continue
            s = cross > 0
            if sign is None:
                sign = s
            elif s != sign:
                return False
        return True

    @staticmethod
    def _seg_dist2(ax, ay, bx, by, px, py):
        """Squared distance from (px,py) to segment (ax,ay)-(bx,by)."""
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom < 1e-12:
            return (px - ax) ** 2 + (py - ay) ** 2
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        return (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2

    def _hit_test(self, mx, my):
        # Fixed screen-space roll controls take priority.
        if my < 30:
            if mx < 35:
                return "roll", -90.0
            if mx > self.SIZE - 35:
                return "roll", 90.0

        projected = self._project()
        visibility = self._face_visibility()

        # A corner or edge is eligible when at least one adjacent face is
        # visible. This preserves silhouette targets without exposing handles
        # from the hidden side of the cube.
        candidates = []
        for index, (vertex, _direction, _up) in enumerate(self._CORNERS):
            adjacent = self._adjacent_faces((vertex,))
            if max(visibility[adjacent]) <= 1e-6:
                continue
            distance = (mx - projected[vertex, 0]) ** 2 + (
                my - projected[vertex, 1]
            ) ** 2
            if distance <= 49.0:
                candidates.append(
                    (distance, -projected[vertex, 2], "corner", index)
                )
        if candidates:
            _, _, kind, index = min(candidates)
            return kind, index

        candidates = []
        for index, (v0, v1, _direction, _up) in enumerate(self._EDGES):
            adjacent = self._adjacent_faces((v0, v1))
            if not adjacent or max(visibility[adjacent]) <= 1e-6:
                continue
            distance = self._seg_dist2(
                projected[v0, 0],
                projected[v0, 1],
                projected[v1, 0],
                projected[v1, 1],
                mx,
                my,
            )
            if distance <= 25.0:
                depth = 0.5 * (projected[v0, 2] + projected[v1, 2])
                candidates.append((distance, -depth, "edge", index))
        if candidates:
            _, _, kind, index = min(candidates)
            return kind, index

        # The most front-facing face wins on shared boundaries.
        for face_index in np.argsort(-visibility):
            if visibility[face_index] <= 1e-6:
                continue
            vertices = self._FACES[face_index][0]
            polygon = [
                (projected[vertex, 0], projected[vertex, 1])
                for vertex in vertices
            ]
            if self._pt_in_poly(polygon, mx, my):
                return "face", int(face_index)
        return None

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):
        hit = self._hit_test(event.x(), event.y())
        if hit != self._hovered:
            self._hovered = hit
            self.setCursor(
                QtCore.Qt.PointingHandCursor if hit else QtCore.Qt.ArrowCursor
            )
            self.update()

    def leaveEvent(self, event):
        self._hovered = None
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        hit = self._hit_test(event.x(), event.y())
        if hit is None:
            return
        kind, index = hit
        if kind == "roll":
            self.roll_requested.emit(index)
            return
        if kind == "face":
            direction = self._FACES[index][2]
        elif kind == "edge":
            direction = self._EDGES[index][2]
        else:
            direction = self._CORNERS[index][1]
        self.view_requested.emit(direction, self._view_up_for(direction))

    # ------------------------------------------------------------------
    # paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        del event
        projected = self._project()
        visibility = self._face_visibility()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        if self._light_mode:
            control = QtGui.QColor(232, 237, 242, 238)
            control_outline = QtGui.QColor(105, 119, 138, 170)
            control_text = QtGui.QColor(63, 76, 94)
            outline = QtGui.QColor(64, 77, 95, 225)
            label_color = QtGui.QColor(255, 255, 255)
        else:
            control = QtGui.QColor(43, 51, 64, 242)
            control_outline = QtGui.QColor(170, 190, 217, 150)
            control_text = QtGui.QColor(205, 219, 238)
            outline = QtGui.QColor(225, 234, 247, 225)
            label_color = QtGui.QColor(255, 255, 255)

        # Fixed roll controls stay readable while the cube follows the camera.
        for rectangle, key in (
            (QtCore.QRectF(6, 3, 28, 28), ("roll", -90.0)),
            (
                QtCore.QRectF(self.SIZE - 34, 3, 28, 28),
                ("roll", 90.0),
            ),
        ):
            hovered = self._hovered == key
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(48, 156, 245)
                    if hovered
                    else control_outline,
                    1.0,
                )
            )
            painter.setBrush(
                QtGui.QColor(42, 132, 207) if hovered else control
            )
            painter.drawRoundedRect(rectangle, 6, 6)

        # Muted axis hues retain the +/- XYZ engineering convention without
        # looking like saturated primary-colour plastic.
        face_colors = (
            np.array((98, 123, 154), dtype=float),  # +Z
            np.array((72, 91, 116), dtype=float),  # -Z
            np.array((91, 130, 111), dtype=float),  # -Y
            np.array((68, 102, 87), dtype=float),  # +Y
            np.array((151, 112, 95), dtype=float),  # +X
            np.array((119, 87, 76), dtype=float),  # -X
        )

        face_depths = [
            float(np.mean(projected[list(face[0]), 2])) for face in self._FACES
        ]
        for face_index in sorted(range(6), key=lambda item: face_depths[item]):
            facing = float(visibility[face_index])
            if facing <= 1e-6:
                continue
            vertices, label, _direction, _up = self._FACES[face_index]
            polygon = QtGui.QPolygonF(
                [
                    QtCore.QPointF(
                        projected[vertex, 0], projected[vertex, 1]
                    )
                    for vertex in vertices
                ]
            )
            hovered = self._hovered == ("face", face_index)
            if hovered:
                color = QtGui.QColor(48, 145, 225)
            else:
                brightness = 0.78 + 0.22 * facing
                rgb = np.clip(
                    face_colors[face_index] * brightness, 0, 255
                ).astype(int)
                color = QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))

            painter.setPen(QtGui.QPen(outline, 1.15))
            painter.setBrush(color)
            painter.drawPolygon(polygon)

            if facing > 0.10:
                painter.setFont(
                    QtGui.QFont("Segoe UI", 7, QtGui.QFont.DemiBold)
                )
                painter.setPen(label_color)
                painter.drawText(
                    polygon.boundingRect(), QtCore.Qt.AlignCenter, label
                )

        # Paint every silhouette/shared edge bordering a visible face. Hidden
        # geometry stays absent, so the cube reads as a solid object.
        for edge_index, (v0, v1, _direction, _up) in enumerate(self._EDGES):
            adjacent = self._adjacent_faces((v0, v1))
            if not adjacent or max(visibility[adjacent]) <= 1e-6:
                continue
            hovered = self._hovered == ("edge", edge_index)
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(48, 156, 245) if hovered else outline,
                    4.0 if hovered else 1.15,
                    QtCore.Qt.SolidLine,
                    QtCore.Qt.RoundCap,
                )
            )
            painter.drawLine(
                QtCore.QPointF(projected[v0, 0], projected[v0, 1]),
                QtCore.QPointF(projected[v1, 0], projected[v1, 1]),
            )

        if self._hovered is not None and self._hovered[0] == "corner":
            vertex = self._CORNERS[self._hovered[1]][0]
            painter.setPen(QtGui.QPen(QtGui.QColor(230, 246, 255), 1.0))
            painter.setBrush(QtGui.QColor(48, 156, 245))
            painter.drawEllipse(
                QtCore.QPointF(
                    projected[vertex, 0], projected[vertex, 1]
                ),
                5.5,
                5.5,
            )

        painter.setFont(QtGui.QFont("Segoe UI Symbol", 15, QtGui.QFont.DemiBold))
        for rectangle, key, symbol in (
            (QtCore.QRectF(7, 4, 26, 26), ("roll", -90.0), "↶"),
            (
                QtCore.QRectF(self.SIZE - 33, 4, 26, 26),
                ("roll", 90.0),
                "↷",
            ),
        ):
            painter.setPen(
                QtGui.QColor(255, 255, 255)
                if self._hovered == key
                else control_text
            )
            painter.drawText(rectangle, QtCore.Qt.AlignCenter, symbol)

        painter.end()
