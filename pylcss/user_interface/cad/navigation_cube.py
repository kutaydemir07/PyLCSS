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

    def set_light_mode(self, enabled):
        self._light_mode = bool(enabled)
        self.update()

    # ── public ────────────────────────────────────────────────────────────

    def update_rotation(self, camera):
        """Extract VTK camera orientation and repaint the cube."""
        pos = np.array(camera.GetPosition(), dtype=float)
        focal = np.array(camera.GetFocalPoint(), dtype=float)
        up = np.array(camera.GetViewUp(), dtype=float)
        fwd = focal - pos
        n = np.linalg.norm(fwd)
        if n < 1e-10:
            return
        fwd /= n
        right = np.cross(fwd, up)
        rn = np.linalg.norm(right)
        if rn < 1e-10:
            return
        right /= rn
        up = np.cross(right, fwd)
        self._rot = np.array([right, up, -fwd])  # rows = camera X/Y/Z in world
        self.update()

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
        # Roll buttons (top-left, top-right corners of widget)
        if my < 30:
            if mx < 35:
                return ("roll", -90.0)
            if mx > self.SIZE - 35:
                return ("roll", 90.0)

        p = self._project()
        vd = self._rot[2]  # camera Z = direction from focal point toward camera
        vis = self._V @ vd  # per-vertex visibility: > 0 means on the viewer side

        # Corners (highest priority).  Only front-facing corners are eligible,
        # and when several fall within the hit radius we keep the one NEAREST
        # THE VIEWER (largest projected depth p[:,2]) so an occluded corner can
        # never win.  Radius tightened to 6 px so the faces stay easy to click.
        best_corner = None
        best_corner_depth = -1e30
        for i, (vi, _, _) in enumerate(self._CORNERS):
            if vis[vi] <= 1e-6:
                continue
            d2 = (mx - p[vi, 0]) ** 2 + (my - p[vi, 1]) ** 2
            if d2 < 36.0 and p[vi, 2] > best_corner_depth:
                best_corner_depth = p[vi, 2]
                best_corner = i
        if best_corner is not None:
            return ("corner", best_corner)

        # Edges: only FULLY visible edges (both endpoints front-facing) are
        # eligible — this drops the depth edges that run across a visible face
        # toward the hidden back corner, which were the ones stealing clicks and
        # navigating to the wrong side.  Ties broken by nearest-the-viewer depth.
        best_edge = None
        best_edge_depth = -1e30
        for i, (v0, v1, _, _) in enumerate(self._EDGES):
            if vis[v0] <= 1e-6 or vis[v1] <= 1e-6:
                continue
            d2 = self._seg_dist2(p[v0, 0], p[v0, 1], p[v1, 0], p[v1, 1], mx, my)
            if d2 < 25.0:
                depth = 0.5 * (p[v0, 2] + p[v1, 2])
                if depth > best_edge_depth:
                    best_edge_depth = depth
                    best_edge = i
        if best_edge is not None:
            return ("edge", best_edge)
        # faces (front-facing first)
        order = sorted(range(6), key=lambda fi: -float(np.dot(self._FACE_N[fi], vd)))
        for fi in order:
            if float(np.dot(self._FACE_N[fi], vd)) < 0.05:
                continue
            vi = self._FACES[fi][0]
            poly = [(p[v, 0], p[v, 1]) for v in vi]
            if self._pt_in_poly(poly, mx, my):
                return ("face", fi)
        return None

    # ── events ────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event):
        hit = self._hit_test(event.x(), event.y())
        if hit != self._hovered:
            self._hovered = hit
            self.update()

    def leaveEvent(self, event):
        if self._hovered is not None:
            self._hovered = None
            self.update()

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        hit = self._hit_test(event.x(), event.y())
        if hit is None:
            return
        kind, idx = hit
        if kind == "roll":
            self.roll_requested.emit(idx)
            return
        if kind == "face":
            _, _, cp, cu = self._FACES[idx]
        elif kind == "edge":
            _, _, cp, cu = self._EDGES[idx]
        else:
            _, cp, cu = self._CORNERS[idx]
        self.view_requested.emit(cp, cu)

    # ── paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = self._project()
        vd = self._rot[2]
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Solid background matching the active VTK viewer theme.
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(
            QtGui.QColor(232, 237, 242)
            if self._light_mode
            else QtGui.QColor(51, 51, 51)
        )
        painter.drawRoundedRect(0, 0, self.SIZE, self.SIZE, 8, 8)

        # Per-face base colors — muted, desaturated "slate" tones that still
        # carry a faint axis hue (Z cool-blue, Y teal-green, X terracotta) so
        # the orientation reads without looking like primary-colour plastic.
        _BASE = [
            QtGui.QColor(96, 120, 150),  # TOP (+Z)   – cool slate blue
            QtGui.QColor(70, 90, 114),  # BOT (-Z)   – dim slate blue
            QtGui.QColor(92, 130, 112),  # FRONT (-Y) – muted teal
            QtGui.QColor(70, 102, 88),  # BACK (+Y)  – dim teal
            QtGui.QColor(150, 112, 96),  # RIGHT (+X) – muted terracotta
            QtGui.QColor(120, 88, 76),  # LEFT (-X)  – dim terracotta
        ]

        # Faces – back-to-front depth sort
        face_depths = [
            float(np.mean([p[vi, 2] for vi in self._FACES[fi][0]])) for fi in range(6)
        ]
        for fi in sorted(range(6), key=lambda fi: face_depths[fi]):
            vis = float(np.dot(self._FACE_N[fi], vd))
            vi_list, label, _, _ = self._FACES[fi]
            poly = [QtCore.QPointF(p[v, 0], p[v, 1]) for v in vi_list]
            qpoly = QtGui.QPolygonF(poly)
            is_hov = self._hovered == ("face", fi)
            base = _BASE[fi]
            if vis > 0:
                # front-facing: full brightness, hover = lighter
                factor = 1.4 if is_hov else 1.0
                col = QtGui.QColor(
                    min(255, int(base.red() * factor)),
                    min(255, int(base.green() * factor)),
                    min(255, int(base.blue() * factor)),
                    255,
                )
                outline = (
                    QtGui.QColor(75, 85, 98, 190)
                    if self._light_mode
                    else QtGui.QColor(220, 230, 255, 180)
                )
                painter.setPen(QtGui.QPen(outline, 1.0))
            else:
                # back-facing: solid dark version
                col = QtGui.QColor(
                    max(0, int(base.red() * 0.28)),
                    max(0, int(base.green() * 0.28)),
                    max(0, int(base.blue() * 0.28)),
                    255,
                )
                painter.setPen(QtGui.QPen(QtGui.QColor(60, 70, 90, 160), 0.6))
            painter.setBrush(col)
            painter.drawPolygon(qpoly)
            if vis > 0.15:
                cx_f = sum(pt.x() for pt in poly) / 4
                cy_f = sum(pt.y() for pt in poly) / 4
                font = QtGui.QFont("Arial", 7 if is_hov else 6, QtGui.QFont.Bold)
                painter.setFont(font)
                painter.setPen(QtGui.QColor(255, 255, 255))
                painter.drawText(
                    QtCore.QRectF(cx_f - 18, cy_f - 8, 36, 16),
                    QtCore.Qt.AlignCenter,
                    label,
                )

        # Edges — draw only FULLY VISIBLE edges (both endpoints on the viewer
        # side) so the cube reads as a solid object, not a see-through
        # wireframe.  The hovered edge gets a bright accent.
        vis = self._V @ vd
        for i, (v0, v1, _, _) in enumerate(self._EDGES):
            if vis[v0] <= 1e-6 or vis[v1] <= 1e-6:
                continue
            is_hov = self._hovered == ("edge", i)
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(30, 105, 170)
                    if is_hov
                    else (
                        QtGui.QColor(80, 90, 105)
                        if self._light_mode
                        else QtGui.QColor(40, 46, 56)
                    ),
                    3.0 if is_hov else 1.0,
                )
            )
            painter.drawLine(
                QtCore.QPointF(p[v0, 0], p[v0, 1]), QtCore.QPointF(p[v1, 0], p[v1, 1])
            )

        # Corners — only the hovered corner is drawn, as a subtle accent dot.
        # Permanent dots on every corner were what made the cube look fake.
        if self._hovered is not None and self._hovered[0] == "corner":
            vi = self._CORNERS[self._hovered[1]][0]
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(150, 205, 255))
            painter.drawEllipse(QtCore.QPointF(p[vi, 0], p[vi, 1]), 5.0, 5.0)

        # Roll Buttons (2D overlays in the top corners)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        font = QtGui.QFont("Arial", 16, QtGui.QFont.Bold)
        painter.setFont(font)

        # Left roll (-90)
        is_hov_l = self._hovered == ("roll", -90.0)
        painter.setPen(
            QtGui.QColor(30, 105, 170)
            if is_hov_l
            else (
                QtGui.QColor(75, 85, 98)
                if self._light_mode
                else QtGui.QColor(120, 140, 160)
            )
        )
        painter.drawText(QtCore.QRectF(8, 5, 25, 25), QtCore.Qt.AlignCenter, "↶")

        # Right roll (+90)
        is_hov_r = self._hovered == ("roll", 90.0)
        painter.setPen(
            QtGui.QColor(30, 105, 170)
            if is_hov_r
            else (
                QtGui.QColor(75, 85, 98)
                if self._light_mode
                else QtGui.QColor(120, 140, 160)
            )
        )
        painter.drawText(
            QtCore.QRectF(self.SIZE - 33, 5, 25, 25), QtCore.Qt.AlignCenter, "↷"
        )

        painter.end()
