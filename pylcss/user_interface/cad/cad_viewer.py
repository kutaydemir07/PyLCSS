# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import vtk
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class NavCubeWidget(QtWidgets.QWidget):
    """Clickable orientation cube overlay.
    Mirrors the scene camera orientation; click any face, edge or corner
    to jump to that standard view.
    """

    view_requested = QtCore.Signal(object, object)  # (pos_tuple, up_tuple)
    roll_requested = QtCore.Signal(float)           # (angle_in_degrees)

    SIZE = 140  # widget pixel size

    # Unit cube vertices  (+X=Right, -Y=Front, +Z=Top, matching viewer axes)
    _V = np.array([
        [-1, -1, -1],  # 0  FLB
        [ 1, -1, -1],  # 1  FRB
        [ 1,  1, -1],  # 2  BRB
        [-1,  1, -1],  # 3  BLB
        [-1, -1,  1],  # 4  FLT
        [ 1, -1,  1],  # 5  FRT
        [ 1,  1,  1],  # 6  BRT
        [-1,  1,  1],  # 7  BLT
    ], dtype=float)

    # faces  (vertex_indices, label, cam_pos_norm, cam_up)
    _FACES = [
        ((4, 5, 6, 7), "TOP",   ( 0,  0,  1), (0,  1,  0)),
        ((3, 2, 1, 0), "BOT",   ( 0,  0, -1), (0,  1,  0)),
        ((0, 1, 5, 4), "FRONT", ( 0, -1,  0), (0,  0,  1)),
        ((2, 3, 7, 6), "BACK",  ( 0,  1,  0), (0,  0,  1)),
        ((1, 2, 6, 5), "RIGHT", ( 1,  0,  0), (0,  0,  1)),
        ((3, 0, 4, 7), "LEFT",  (-1,  0,  0), (0,  0,  1)),
    ]
    _FACE_N = np.array([
        ( 0,  0,  1), ( 0,  0, -1), ( 0, -1,  0),
        ( 0,  1,  0), ( 1,  0,  0), (-1,  0,  0),
    ], dtype=float)

    # edges  (v0, v1, cam_pos_norm, cam_up)
    _EDGES = [
        (4, 5, ( 0, -1,  1), (0,  0,  1)),   # FT
        (5, 6, ( 1,  0,  1), (0,  0,  1)),   # RT
        (6, 7, ( 0,  1,  1), (0,  0,  1)),   # BT
        (7, 4, (-1,  0,  1), (0,  0,  1)),   # LT
        (0, 1, ( 0, -1, -1), (0,  0, -1)),   # FB
        (1, 2, ( 1,  0, -1), (0,  0, -1)),   # RB
        (2, 3, ( 0,  1, -1), (0,  0, -1)),   # BB
        (3, 0, (-1,  0, -1), (0,  0, -1)),   # LB
        (4, 0, (-1, -1,  0), (0,  0,  1)),   # FL
        (5, 1, ( 1, -1,  0), (0,  0,  1)),   # FR
        (6, 2, ( 1,  1,  0), (0,  0,  1)),   # BR
        (7, 3, (-1,  1,  0), (0,  0,  1)),   # BL
    ]

    # corners  (vertex_idx, cam_pos_norm, cam_up)
    _CORNERS = [
        (0, (-1, -1, -1), (0,  0, -1)),  # FLB
        (1, ( 1, -1, -1), (0,  0, -1)),  # FRB
        (2, ( 1,  1, -1), (0,  0, -1)),  # BRB
        (3, (-1,  1, -1), (0,  0, -1)),  # BLB
        (4, (-1, -1,  1), (0,  0,  1)),  # FLT
        (5, ( 1, -1,  1), (0,  0,  1)),  # FRT
        (6, ( 1,  1,  1), (0,  0,  1)),  # BRT
        (7, (-1,  1,  1), (0,  0,  1)),  # BLT
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self.setAutoFillBackground(False)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._rot = np.eye(3)
        self._hovered = None

    # ── public ────────────────────────────────────────────────────────────

    def update_rotation(self, camera):
        """Extract VTK camera orientation and repaint the cube."""
        pos   = np.array(camera.GetPosition(),   dtype=float)
        focal = np.array(camera.GetFocalPoint(), dtype=float)
        up    = np.array(camera.GetViewUp(),     dtype=float)
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
        self._rot = np.array([right, up, -fwd])   # rows = camera X/Y/Z in world
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
            if mx < 35: return ('roll', -90.0)
            if mx > self.SIZE - 35: return ('roll', 90.0)

        p  = self._project()
        vd = self._rot[2]  # camera Z = direction we're looking along
        # corners (highest priority)
        for i, (vi, _, _) in enumerate(self._CORNERS):
            if (mx - p[vi, 0]) ** 2 + (my - p[vi, 1]) ** 2 < 64:
                return ('corner', i)
        # edges
        for i, (v0, v1, _, _) in enumerate(self._EDGES):
            if self._seg_dist2(p[v0,0], p[v0,1], p[v1,0], p[v1,1], mx, my) < 36:
                return ('edge', i)
        # faces (front-facing first)
        order = sorted(range(6), key=lambda fi: -float(np.dot(self._FACE_N[fi], vd)))
        for fi in order:
            if float(np.dot(self._FACE_N[fi], vd)) < 0.05:
                continue
            vi = self._FACES[fi][0]
            poly = [(p[v, 0], p[v, 1]) for v in vi]
            if self._pt_in_poly(poly, mx, my):
                return ('face', fi)
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
        if kind == 'roll':
            self.roll_requested.emit(idx)
            return
        if kind == 'face':
            _, _, cp, cu = self._FACES[idx]
        elif kind == 'edge':
            _, _, cp, cu = self._EDGES[idx]
        else:
            _, cp, cu = self._CORNERS[idx]
        self.view_requested.emit(cp, cu)

    # ── paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p  = self._project()
        vd = self._rot[2]
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Solid background matching VTK renderer bg (0.2,0.2,0.2) = rgb(51,51,51)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(51, 51, 51))
        painter.drawRoundedRect(0, 0, self.SIZE, self.SIZE, 8, 8)

        # Per-face base colors (front-face tone; back-face dims automatically)
        _BASE = [
            QtGui.QColor( 50, 110, 200),  # TOP   – blue
            QtGui.QColor( 35,  80, 150),  # BOT   – darker blue
            QtGui.QColor( 45, 170,  90),  # FRONT – green
            QtGui.QColor( 35, 130,  70),  # BACK  – darker green
            QtGui.QColor(210,  90,  50),  # RIGHT – orange-red
            QtGui.QColor(160,  70,  40),  # LEFT  – darker orange
        ]

        # Faces – back-to-front depth sort
        face_depths = [float(np.mean([p[vi, 2] for vi in self._FACES[fi][0]])) for fi in range(6)]
        for fi in sorted(range(6), key=lambda fi: face_depths[fi]):
            vis = float(np.dot(self._FACE_N[fi], vd))
            vi_list, label, _, _ = self._FACES[fi]
            poly = [QtCore.QPointF(p[v, 0], p[v, 1]) for v in vi_list]
            qpoly = QtGui.QPolygonF(poly)
            is_hov = self._hovered == ('face', fi)
            base = _BASE[fi]
            if vis > 0:
                # front-facing: full brightness, hover = lighter
                factor = 1.4 if is_hov else 1.0
                col = QtGui.QColor(
                    min(255, int(base.red()   * factor)),
                    min(255, int(base.green() * factor)),
                    min(255, int(base.blue()  * factor)),
                    255,
                )
                painter.setPen(QtGui.QPen(QtGui.QColor(220, 230, 255, 180), 1.0))
            else:
                # back-facing: solid dark version
                col = QtGui.QColor(
                    max(0, int(base.red()   * 0.28)),
                    max(0, int(base.green() * 0.28)),
                    max(0, int(base.blue()  * 0.28)),
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
                painter.drawText(QtCore.QRectF(cx_f - 18, cy_f - 8, 36, 16),
                                 QtCore.Qt.AlignCenter, label)

        # Edges
        for i, (v0, v1, _, _) in enumerate(self._EDGES):
            is_hov = self._hovered == ('edge', i)
            painter.setPen(QtGui.QPen(
                QtGui.QColor(140, 200, 255) if is_hov else QtGui.QColor(90, 120, 165),
                3.5 if is_hov else 1.2
            ))
            painter.drawLine(QtCore.QPointF(p[v0, 0], p[v0, 1]),
                             QtCore.QPointF(p[v1, 0], p[v1, 1]))

        # Corners
        painter.setPen(QtCore.Qt.NoPen)
        for i, (vi, _, _) in enumerate(self._CORNERS):
            is_hov = self._hovered == ('corner', i)
            r = 5.5 if is_hov else 3.5
            painter.setBrush(QtGui.QColor(105, 190, 255) if is_hov else QtGui.QColor(68, 108, 162))
            painter.drawEllipse(QtCore.QPointF(p[vi, 0], p[vi, 1]), r, r)

        # Roll Buttons (2D overlays in the top corners)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        font = QtGui.QFont("Arial", 16, QtGui.QFont.Bold)
        painter.setFont(font)
        
        # Left roll (-90)
        is_hov_l = self._hovered == ('roll', -90.0)
        painter.setPen(QtGui.QColor(200, 230, 255) if is_hov_l else QtGui.QColor(120, 140, 160))
        painter.drawText(QtCore.QRectF(8, 5, 25, 25), QtCore.Qt.AlignCenter, "↶")
        
        # Right roll (+90)
        is_hov_r = self._hovered == ('roll', 90.0)
        painter.setPen(QtGui.QColor(200, 230, 255) if is_hov_r else QtGui.QColor(120, 140, 160))
        painter.drawText(QtCore.QRectF(self.SIZE - 33, 5, 25, 25), QtCore.Qt.AlignCenter, "↷")

        painter.end()


class CQ3DViewer(QtWidgets.QWidget):
    """
    Professional 3D Viewer for CadQuery using VTK.
    Embeds directly into PySide6 layouts.
    Supports interactive face picking mode for FEA boundary condition assignment.
    """

    # Signals emitted during interactive picking
    face_picked = QtCore.Signal(list)      # list of OCC face objects
    picking_cancelled = QtCore.Signal()
    face_picking_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super(CQ3DViewer, self).__init__(parent)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- View Orientation Toolbar ---
        self._view_toolbar = QtWidgets.QWidget(self)
        self._view_toolbar.setObjectName("view_toolbar")
        self._view_toolbar.setStyleSheet(
            "#view_toolbar { background: rgba(30, 30, 30, 180); border-bottom: 1px solid #444; }"
            "QPushButton { background: transparent; color: #ccc; font-weight: bold; border-radius: 3px; padding: 4px 10px; }"
            "QPushButton:hover { background: rgba(80, 80, 80, 200); color: white; }"
        )
        vtb_layout = QtWidgets.QHBoxLayout(self._view_toolbar)
        vtb_layout.setContentsMargins(5, 2, 5, 2)
        vtb_layout.addStretch()

        self._btn_pick_faces = QtWidgets.QPushButton("Pick Faces")
        self._btn_pick_faces.setToolTip(
            "Start face picking for the selected Select Face (Interactive) node"
        )
        self._btn_pick_faces.clicked.connect(self.face_picking_requested.emit)
        vtb_layout.addWidget(self._btn_pick_faces)

        btn_grid = QtWidgets.QPushButton("Grid")
        btn_grid.setCheckable(True)
        btn_grid.setChecked(False)
        btn_grid.setStyleSheet(
            "QPushButton { background: transparent; color: #ccc; font-weight: bold; border-radius: 3px; padding: 4px 10px; }"
            "QPushButton:hover { background: rgba(80, 80, 80, 200); color: white; }"
            "QPushButton:checked { background: rgba(74, 158, 255, 100); color: #4a9eff; border: 1px solid #4a9eff; }"
        )
        btn_grid.clicked.connect(self._toggle_grid)
        vtb_layout.addWidget(btn_grid)

        self.main_layout.addWidget(self._view_toolbar)

        # --- Picking toolbar (hidden by default) ---
        self._picking_toolbar = QtWidgets.QWidget(self)
        self._picking_toolbar.setObjectName("picking_toolbar")
        self._picking_toolbar.setStyleSheet(
            "#picking_toolbar {"
            "  background: rgba(30, 90, 180, 220);"
            "  border-bottom: 2px solid #4a9eff;"
            "}"
        )
        tb_layout = QtWidgets.QHBoxLayout(self._picking_toolbar)
        tb_layout.setContentsMargins(12, 6, 12, 6)

        self._pick_icon_lbl = QtWidgets.QLabel("Face Picking Mode  --  Click faces to select")
        self._pick_icon_lbl.setStyleSheet("color: white; font-weight: bold; font-size: 13px;")
        tb_layout.addWidget(self._pick_icon_lbl)

        tb_layout.addStretch()

        self._pick_count_lbl = QtWidgets.QLabel("0 selected")
        self._pick_count_lbl.setStyleSheet("color: #aad4ff; font-size: 12px; margin-right: 16px;")
        tb_layout.addWidget(self._pick_count_lbl)

        self._pick_hint_lbl = QtWidgets.QLabel("Ctrl+Click = multi-select")
        self._pick_hint_lbl.setStyleSheet("color: #aad4ff; font-size: 11px; margin-right: 16px;")
        tb_layout.addWidget(self._pick_hint_lbl)

        btn_done = QtWidgets.QPushButton("Done")
        btn_done.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; border-radius: 4px;"
            "  padding: 4px 14px; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #2ecc71; }"
        )
        btn_done.clicked.connect(self._on_pick_done)
        tb_layout.addWidget(btn_done)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; border-radius: 4px;"
            "  padding: 4px 14px; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #e74c3c; }"
        )
        btn_cancel.clicked.connect(self._on_pick_cancel)
        tb_layout.addWidget(btn_cancel)

        self._picking_toolbar.hide()
        self.main_layout.addWidget(self._picking_toolbar)

        # VTK Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.main_layout.addWidget(self.vtkWidget)

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

        # Initialize sophisticated View Axes
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.02)
        axes.SetConeRadius(0.08)
        
        # Style the labels
        for txt in [axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(), axes.GetZAxisCaptionActor2D()]:
            txt.GetCaptionTextProperty().SetColor(1, 1, 1)
            txt.GetCaptionTextProperty().SetFontFamilyToArial()
            txt.GetCaptionTextProperty().BoldOn()
            txt.GetCaptionTextProperty().ItalicOff()
            txt.GetCaptionTextProperty().ShadowOff()
            txt.GetCaptionTextProperty().SetFontSize(24)
            txt.SetWidth(0.1)

        self.marker_widget = vtk.vtkOrientationMarkerWidget()
        self.marker_widget.SetOrientationMarker(axes)
        self.marker_widget.SetInteractor(self.interactor)
        self.marker_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.marker_widget.SetEnabled(1)
        self.marker_widget.InteractiveOff()  # Prevent user from dragging it around
        
        # Initialize Grid state
        self._grid_actor = None
        self._axes_actor = None

        # State
        self.current_actor = None
        self.actors = []  # List of all active actors

        # --- Picking State ---
        self._picking_mode = False
        self._multi_select = True
        self._picked_face_indices = []       # list of int (OCC face indices)
        self._picked_occ_faces = []          # list of OCC face objects
        self._highlight_actors = []          # VTK actors for selected face highlights
        self._face_map = {}                  # vtk_cell_id -> occ_face_index
        self._all_occ_faces = []             # list of OCC face objects (from last render_shape)
        self._face_polydata_list = []        # per-face vtkPolyData for highlighting
        self._pick_callback_id = None

        # --- BC Overlay State ---
        self._bc_overlay_actors = []     # dedicated list for load/support overlay actors
        self._cached_bc_data = None      # (constraint_faces, load_faces, load_vectors) – replayed after sim render

        # --- Crash Playback State ---
        self._crash_frames      = []         # list of frame dicts
        self._crash_base_data   = None       # original crash result (mesh, viz_mode, etc.)
        self._crash_frame_idx   = 0          # current frame index
        self._crash_playing     = False
        self._crash_scalar_range = (0.0, 1.0)  # global min/max for stable colourmap
        self._crash_timer       = QtCore.QTimer(self)
        self._crash_timer.timeout.connect(self._on_crash_timer)

        # Crash panel (hidden until crash results arrive)
        self._crash_panel = self._build_crash_panel()
        self._crash_panel.hide()
        self.main_layout.addWidget(self._crash_panel)

        self.interactor.Initialize()
        self.interactor.Start()

        # NavCube overlay – replaces the VTK axes orientation marker widget
        self.marker_widget.SetEnabled(0)
        self._nav_cube = NavCubeWidget(self)
        self._nav_cube.view_requested.connect(self._set_camera_view)
        self._nav_cube.roll_requested.connect(self._roll_camera)
        self._nav_cube.raise_()
        self._nav_cube.show()
        self._position_nav_cube()
        self.vtkWidget.GetRenderWindow().AddObserver("EndEvent", lambda o, e: self._on_vtk_render())

    # ──────────────────────────────────────────────────────────────────────────
    # CRASH ANIMATION PLAYBACK
    # ──────────────────────────────────────────────────────────────────────────

    def _build_crash_panel(self):
        """Construct the crash animation playback toolbar (initially hidden)."""
        panel = QtWidgets.QWidget(self)
        panel.setObjectName("crash_panel")
        panel.setFixedHeight(52)
        panel.setStyleSheet(
            "#crash_panel {"
            "  background: rgba(15, 20, 35, 230);"
            "  border-top: 2px solid #e74c3c;"
            "}"
            "QLabel { color: #ecf0f1; font-size: 11px; }"
            "QPushButton {"
            "  background: #2c3e50; color: white; border: 1px solid #4a4a6a;"
            "  border-radius: 4px; padding: 3px 10px; font-size: 12px;"
            "}"
            "QPushButton:hover { background: #e74c3c; }"
            "QSlider::groove:horizontal { height: 4px; background: #4a4a6a; border-radius: 2px; }"
            "QSlider::handle:horizontal {"
            "  background: #e74c3c; width: 14px; height: 14px;"
            "  border-radius: 7px; margin: -5px 0;"
            "}"
            "QSlider::sub-page:horizontal { background: #e74c3c; border-radius: 2px; }"
            "QComboBox {"
            "  background: #2c3e50; color: white; border: 1px solid #4a4a6a;"
            "  border-radius: 3px; padding: 2px 6px; font-size: 11px;"
            "}"
        )

        lay = QtWidgets.QHBoxLayout(panel)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(8)

        # Crash icon label
        icon_lbl = QtWidgets.QLabel("   CRASH PLAYBACK")
        icon_lbl.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 12px;")
        lay.addWidget(icon_lbl)

        # Rewind button
        btn_rew = QtWidgets.QPushButton("\u23EE")   # ⏮
        btn_rew.setFixedWidth(32)
        btn_rew.setToolTip("Rewind to start")
        btn_rew.clicked.connect(self._crash_rewind)
        lay.addWidget(btn_rew)

        # Play/Pause button
        self._play_btn = QtWidgets.QPushButton("\u25B6")   # ▶
        self._play_btn.setFixedWidth(32)
        self._play_btn.setToolTip("Play / Pause (Space)")
        self._play_btn.clicked.connect(self._toggle_crash_play)
        lay.addWidget(self._play_btn)

        # Slider
        self._crash_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._crash_slider.setMinimum(0)
        self._crash_slider.setMaximum(0)
        self._crash_slider.setValue(0)
        self._crash_slider.setToolTip("Drag to scrub through time")
        self._crash_slider.valueChanged.connect(self._on_crash_slider_changed)
        lay.addWidget(self._crash_slider, stretch=1)

        # Time label
        self._crash_time_lbl = QtWidgets.QLabel("t = 0.000 ms")
        self._crash_time_lbl.setFixedWidth(95)
        lay.addWidget(self._crash_time_lbl)

        # Speed combo
        lay.addWidget(QtWidgets.QLabel("Speed:"))
        self._speed_combo = QtWidgets.QComboBox()
        for label in ["0.25\u00d7", "0.5\u00d7", "1\u00d7", "2\u00d7", "4\u00d7", "8\u00d7"]:
            self._speed_combo.addItem(label)
        self._speed_combo.setCurrentIndex(2)  # 1×
        self._speed_combo.setFixedWidth(62)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        lay.addWidget(self._speed_combo)

        # Frame counter
        self._crash_frame_lbl = QtWidgets.QLabel("0 / 0")
        self._crash_frame_lbl.setFixedWidth(55)
        lay.addWidget(self._crash_frame_lbl)

        # Close button
        btn_close = QtWidgets.QPushButton("\u2715")  # ✕
        btn_close.setFixedWidth(28)
        btn_close.setToolTip("Close playback panel")
        btn_close.clicked.connect(self.stop_crash_playback)
        lay.addWidget(btn_close)

        return panel

    # ── Public entry point ────────────────────────────────────────────────────

    def start_crash_playback(self, crash_data):
        """
        Load crash result data into the playback panel and begin animating.
        Call this instead of render_simulation() when crash results arrive.
        """
        frames = crash_data.get('frames', [])
        if not frames:
            # No frames recorded — fall back to static render
            self.render_simulation(crash_data)
            return

        self._crash_frames    = frames
        self._crash_base_data = crash_data
        self._crash_frame_idx = 0
        self._crash_playing   = False

        # Compute global scalar range for stable colourmap across all frames
        viz_mode = crash_data.get('visualization_mode', 'Von Mises Stress')
        disp_scale_val = float(crash_data.get('disp_scale', 1.0))
        all_vals = []
        for fr in frames:
            field = self._crash_field_for_frame(fr, viz_mode)
            if field is not None and len(field) > 0:
                # For Displacement mode the global range must reflect the
                # scaled magnitudes that will actually be painted on the mesh.
                range_scale = disp_scale_val if viz_mode == 'Displacement' else 1.0
                all_vals.append(float(np.min(field)) * range_scale)
                all_vals.append(float(np.max(field)) * range_scale)
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            if hi - lo < 1e-10:
                hi = lo + 1.0
            self._crash_scalar_range = (lo, hi)
        else:
            self._crash_scalar_range = (0.0, 1.0)

        # Configure slider
        self._crash_slider.blockSignals(True)
        self._crash_slider.setMaximum(len(frames) - 1)
        self._crash_slider.setValue(0)
        self._crash_slider.blockSignals(False)

        # ── Set camera to encompass the fully-deformed bounding box ──────────
        # Using the last frame (maximum deformation) ensures the whole range
        # of motion is visible from the very first frame rendered, without the
        # camera ever being reset to the undeformed (frame-0) state.
        mesh = crash_data.get('mesh')
        disp_scale = float(crash_data.get('disp_scale', 1.0))
        if mesh is not None and frames:
            last_disp = frames[-1].get('displacement')
            self._set_camera_for_crash(mesh, last_disp, disp_scale)

        # Show panel
        self._crash_panel.show()

        # Render first frame (camera is already positioned — never reset it again)
        self._render_crash_frame(0)
        self._toggle_crash_play()   # start playing

    def stop_crash_playback(self):
        """Stop playback and hide the crash panel."""
        self._crash_timer.stop()
        self._crash_playing = False
        self._play_btn.setText("\u25B6")
        self._crash_panel.hide()
        self._crash_frames      = []
        self._crash_base_data   = None

    def _set_camera_for_crash(self, mesh, last_displacement, disp_scale=1.0):
        """
        Position the camera so that both the original mesh AND the most-deformed
        state are fully visible.  Called once when crash playback starts; the
        camera is then kept fixed for the entire animation (including loops).

        Parameters
        ----------
        mesh            : skfem MeshTet – original (undeformed) mesh
        last_displacement: (3·N,) displacement array for the final frame
        disp_scale       : visualisation scale factor applied to displacement
        """
        p0 = mesh.p                                   # (3, N) original coords
        n  = p0.shape[1]

        # Start from original bounding box
        x0_min, x0_max = float(p0[0].min()), float(p0[0].max())
        y0_min, y0_max = float(p0[1].min()), float(p0[1].max())
        z0_min, z0_max = float(p0[2].min()), float(p0[2].max())

        # Expand to include the final deformed state
        if last_displacement is not None and len(last_displacement) == 3 * n:
            d3 = last_displacement.reshape((3, n), order='F') * disp_scale
            pf = p0 + d3
            x0_min = min(x0_min, float(pf[0].min()))
            x0_max = max(x0_max, float(pf[0].max()))
            y0_min = min(y0_min, float(pf[1].min()))
            y0_max = max(y0_max, float(pf[1].max()))
            z0_min = min(z0_min, float(pf[2].min()))
            z0_max = max(z0_max, float(pf[2].max()))

        # Add 10 % margin on every side
        mx = (x0_max - x0_min) * 0.10
        my = (y0_max - y0_min) * 0.10
        mz = (z0_max - z0_min) * 0.10
        bounds = [
            x0_min - mx, x0_max + mx,
            y0_min - my, y0_max + my,
            z0_min - mz, z0_max + mz,
        ]
        self.renderer.ResetCamera(bounds)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _crash_field_for_frame(frame, viz_mode):
        """Return the correct scalar field from a frame dict for a given viz mode."""
        if viz_mode == 'Von Mises Stress':
            return frame.get('stress_vm')
        elif viz_mode == 'Displacement':
            u = frame.get('displacement')
            if u is not None:
                n = len(u) // 3
                return np.linalg.norm(u.reshape(n, 3), axis=1)
            return None
        elif viz_mode == 'Plastic Strain':
            return frame.get('eps_p')
        elif viz_mode == 'Failed Elements':
            return frame.get('failed')
        return frame.get('stress_vm')

    def _render_crash_frame(self, idx):
        """Render a single crash animation frame at the given index."""
        if not self._crash_frames or not self._crash_base_data:
            return
        idx = max(0, min(idx, len(self._crash_frames) - 1))
        frame = self._crash_frames[idx]

        # Build a view-data dict that re-uses the cached mesh
        viz_mode   = self._crash_base_data.get('visualization_mode', 'Von Mises Stress')
        disp_scale = float(self._crash_base_data.get('disp_scale', 1.0))
        field      = self._crash_field_for_frame(frame, viz_mode)

        # Scale displacement for visualisation (disp_scale > 1 magnifies the
        # deformation so small structural changes become clearly visible).
        raw_disp = frame['displacement']
        vis_disp = raw_disp * disp_scale if disp_scale != 1.0 else raw_disp

        view_data = {
            'type':               'crash_frame',   # avoids re-triggering start_crash_playback
            'mesh':               self._crash_base_data['mesh'],
            'displacement':       vis_disp,
            'stress':             field,
            'visualization_mode': viz_mode,
            '_scalar_range':      self._crash_scalar_range,  # pass locked range
            # Camera is set once in start_crash_playback based on the fully-
            # deformed bounding box — never reset it again during playback so
            # the viewer provides a stable, fixed-viewpoint reference throughout
            # the animation (including subsequent loops).
            '_reset_camera':      False,
        }
        self.render_simulation(view_data)

        # Update UI labels
        t_ms = float(frame.get('time', 0.0))
        self._crash_time_lbl.setText(f"t = {t_ms:.3f} ms")
        self._crash_frame_lbl.setText(f"{idx + 1} / {len(self._crash_frames)}")

        # Keep slider in sync without triggering valueChanged recursion
        self._crash_slider.blockSignals(True)
        self._crash_slider.setValue(idx)
        self._crash_slider.blockSignals(False)

    def _on_crash_timer(self):
        """Timer callback: advance one frame."""
        if not self._crash_frames:
            self._crash_timer.stop()
            return
        next_idx = self._crash_frame_idx + 1
        if next_idx >= len(self._crash_frames):
            # Loop back to start
            next_idx = 0
        self._crash_frame_idx = next_idx
        self._render_crash_frame(next_idx)

    def _toggle_crash_play(self):
        """Toggle play / pause."""
        if self._crash_playing:
            self._crash_timer.stop()
            self._crash_playing = False
            self._play_btn.setText("\u25B6")
        else:
            self._crash_playing = True
            self._play_btn.setText("\u23F8")   # ⏸
            self._update_timer_interval()
            self._crash_timer.start()

    def _crash_rewind(self):
        """Jump to frame 0."""
        self._crash_timer.stop()
        self._crash_playing = False
        self._play_btn.setText("\u25B6")
        self._crash_frame_idx = 0
        self._render_crash_frame(0)

    def _on_crash_slider_changed(self, value):
        """User dragged the slider — pause and jump to that frame."""
        self._crash_timer.stop()
        self._crash_playing = False
        self._play_btn.setText("\u25B6")
        self._crash_frame_idx = value
        self._render_crash_frame(value)

    def _on_speed_changed(self, _):
        """Playback speed changed — update timer interval."""
        if self._crash_playing:
            self._update_timer_interval()
            self._crash_timer.start()

    def _update_timer_interval(self):
        """Set QTimer interval based on speed combo selection."""
        # Target ~30 fps at 1×; speed multiplier makes frames go faster/slower
        speed_map = {0: 0.25, 1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0, 5: 8.0}
        speed = speed_map.get(self._speed_combo.currentIndex(), 1.0)
        interval_ms = max(8, int(33.0 / speed))
        self._crash_timer.setInterval(interval_ms)

    # ──────────────────────────────────────────────────────────────────────────
    # PICKING MODE
    # ──────────────────────────────────────────────────────────────────────────

    def enable_picking_mode(self, multi_select=True):
        """Switch the viewer into face-selection picking mode."""
        self._picking_mode = True
        self._multi_select = multi_select
        self._picked_face_indices = []
        self._picked_occ_faces = []
        self._clear_highlight_actors()

        # Show toolbar
        self._picking_toolbar.show()
        self._pick_count_lbl.setText("0 selected")
        self._btn_pick_faces.setEnabled(False)

        # Install VTK left-button press callback
        self._pick_callback_id = self.interactor.AddObserver(
            "LeftButtonPressEvent", self._on_vtk_pick
        )

        # Change cursor
        self.vtkWidget.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

    def disable_picking_mode(self):
        """Exit picking mode, restore normal orbit interaction."""
        self._picking_mode = False
        self._picking_toolbar.hide()
        self._clear_highlight_actors()

        if self._pick_callback_id is not None:
            try:
                self.interactor.RemoveObserver(self._pick_callback_id)
            except Exception:
                pass
            self._pick_callback_id = None

        self.vtkWidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self._btn_pick_faces.setEnabled(True)

    def _on_vtk_pick(self, obj, event):
        """VTK callback: called on left-button press in picking mode."""
        if not self._picking_mode:
            return

        x, y = self.interactor.GetEventPosition()
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        ctrl_held = bool(modifiers & QtCore.Qt.ControlModifier)

        # Use cell picker for face selection
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.001)
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
            self._picked_occ_faces = [self._all_occ_faces[face_idx]] if face_idx < len(self._all_occ_faces) else []
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

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.55, 0.0)   # Orange highlight
            actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().SetLineWidth(2.0)
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(1.0, 0.8, 0.0)

            self.renderer.AddActor(actor)
            self._highlight_actors.append(actor)

        self.vtkWidget.GetRenderWindow().Render()

    def _clear_highlight_actors(self):
        """Remove all highlight actors from the scene."""
        for actor in self._highlight_actors:
            self.renderer.RemoveActor(actor)
        self._highlight_actors = []

    def _on_pick_done(self):
        """User confirmed picking — emit signal and exit picking mode."""
        picked = list(self._picked_occ_faces)
        picked_indices = list(self._picked_face_indices)
        self.disable_picking_mode()
        # Re-add highlights in a passive/dimmed color to keep them visible
        for idx in picked_indices:
            if idx < len(self._face_polydata_list) and self._face_polydata_list[idx] is not None:
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(self._face_polydata_list[idx])
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.6, 0.1)
                actor.GetProperty().SetOpacity(0.5)
                self.renderer.AddActor(actor)
                self.actors.append(actor)
        self.vtkWidget.GetRenderWindow().Render()
        self.face_picked.emit(picked)

    def _on_pick_cancel(self):
        """User cancelled picking."""
        self.disable_picking_mode()
        self.picking_cancelled.emit()

    def highlight_faces(self, face_indices):
        """Public method to highlight specific face indices matching current geometry."""
        self._clear_highlight_actors()
        for idx in face_indices:
            if idx < len(self._face_polydata_list) and self._face_polydata_list[idx] is not None:
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(self._face_polydata_list[idx])
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.6, 0.1)  # Dim orange
                actor.GetProperty().SetOpacity(0.5)
                self.renderer.AddActor(actor)
                self.actors.append(actor)
        self.vtkWidget.GetRenderWindow().Render()

    # ──────────────────────────────────────────────────────────────────────────
    # CONFIRM / CANCEL from external code
    # ──────────────────────────────────────────────────────────────────────────

    def confirm_picking(self):
        self._on_pick_done()

    def cancel_picking(self):
        self._on_pick_cancel()

    # ──────────────────────────────────────────────────────────────────────────
    # CLEAR
    # ──────────────────────────────────────────────────────────────────────────

    def clear(self):
        """Clear the viewer and release memory."""
        self._clear_highlight_actors()

        # Clear BC overlay actors
        for actor in list(self._bc_overlay_actors):
            self.renderer.RemoveActor(actor)
        self._bc_overlay_actors = []
        # Reset cached BC data so overlays don't auto-replay on wrong shapes
        self._cached_bc_data = None

        # Clear legacy single actor if present
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            if self.current_actor.GetMapper():
                self.current_actor.GetMapper().RemoveAllInputConnections(0)
            self.current_actor = None

        # Clear all actors in the list (excluding bc_overlay which were
        # already removed above to avoid double-removal).
        # NB: vtkBillboardTextActor3D and other text actors do not have a
        #     GetMapper() method, so guard the mapper teardown with hasattr.
        for actor in list(self.actors):
            if getattr(actor, '_bc_overlay', False):
                continue
            self.renderer.RemoveActor(actor)
            get_mapper = getattr(actor, 'GetMapper', None)
            if callable(get_mapper):
                mapper = get_mapper()
                if mapper is not None:
                    try:
                        mapper.RemoveAllInputConnections(0)
                    except Exception:
                        pass
        self.actors = []

        self.scalar_bar.VisibilityOff()
        self.vtkWidget.GetRenderWindow().Render()

    # ── NavCube helpers ───────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_nav_cube'):
            self._position_nav_cube()

    def _position_nav_cube(self):
        """Keep the NavCube anchored to the bottom-left of the VTK area."""
        margin = 8
        y = self.height() - NavCubeWidget.SIZE - margin
        self._nav_cube.move(margin, y)

    def _on_vtk_render(self):
        """Sync NavCube rotation to camera after every VTK render."""
        try:
            self._nav_cube.update_rotation(self.renderer.GetActiveCamera())
        except Exception:
            pass

    def _toggle_grid(self, state):
        """Toggle grid visibility."""
        if state:
            if not self._grid_actor:
                self._grid_actor, self._axes_actor = self._build_grid_actors()
            self.renderer.AddActor(self._grid_actor)
            self.renderer.AddActor(self._axes_actor)
        else:
            if self._grid_actor:
                self.renderer.RemoveActor(self._grid_actor)
                self.renderer.RemoveActor(self._axes_actor)
        self.vtkWidget.GetRenderWindow().Render()

    def _build_grid_actors(self):
        """Creates a 3-plane (XY/XZ/YZ) reference grid and thick main axes for spatial reference."""
        size = 500.0  # Big enough to feel infinite
        step = 10.0
        
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        idx = 0

        def add_line(p0, p1, r, g, b):
            nonlocal idx
            pts.InsertNextPoint(*p0)
            pts.InsertNextPoint(*p1)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(idx)
            lines.InsertCellPoint(idx + 1)
            colors.InsertNextTuple3(r, g, b)
            idx += 2

        ticks = np.arange(-size, size + step, step)

        # XY plane (z=0) — neutral gray
        for v in ticks:
            add_line(( v, -size, 0), ( v,  size, 0), 58, 58, 58)
            add_line((-size,  v, 0), ( size, v,  0), 58, 58, 58)

        # XZ plane (y=0) — faint blue tint
        for v in ticks:
            add_line(( v, 0, -size), ( v, 0,  size), 48, 48, 65)
            add_line((-size, 0,  v), ( size, 0, v),  48, 48, 65)

        # YZ plane (x=0) — faint red tint
        for v in ticks:
            add_line((0,  v, -size), (0,  v,  size), 65, 48, 48)
            add_line((0, -size,  v), (0,  size, v),  65, 48, 48)

        grid_pd = vtk.vtkPolyData()
        grid_pd.SetPoints(pts)
        grid_pd.SetLines(lines)
        grid_pd.GetCellData().SetScalars(colors)
        
        grid_mapper = vtk.vtkPolyDataMapper()
        grid_mapper.SetInputData(grid_pd)
        grid_mapper.SetColorModeToDirectScalars()
        grid_actor = vtk.vtkActor()
        grid_actor.SetMapper(grid_mapper)
        grid_actor.GetProperty().SetLineWidth(1.0)
        grid_actor.GetProperty().LightingOff()
        grid_actor.SetPickable(0)
        grid_actor.UseBoundsOff()   # exclude from ResetCamera bounds

        # Thick Central Axes Crosshair
        axes_pd = vtk.vtkPolyData()
        axes_pts = vtk.vtkPoints()
        axes_lines = vtk.vtkCellArray()
        axes_colors = vtk.vtkUnsignedCharArray()
        axes_colors.SetNumberOfComponents(3)
        
        # X Red
        axes_pts.InsertNextPoint(-size, 0, 0)
        axes_pts.InsertNextPoint(size, 0, 0)
        axes_lines.InsertNextCell(2, [0, 1])
        axes_colors.InsertNextTuple3(255, 50, 50)
        
        # Y Green
        axes_pts.InsertNextPoint(0, -size, 0)
        axes_pts.InsertNextPoint(0, size, 0)
        axes_lines.InsertNextCell(2, [2, 3])
        axes_colors.InsertNextTuple3(50, 255, 50)
        
        # Z Blue
        axes_pts.InsertNextPoint(0, 0, -size)
        axes_pts.InsertNextPoint(0, 0, size)
        axes_lines.InsertNextCell(2, [4, 5])
        axes_colors.InsertNextTuple3(50, 50, 255)

        axes_pd.SetPoints(axes_pts)
        axes_pd.SetLines(axes_lines)
        axes_pd.GetCellData().SetScalars(axes_colors)
        
        axes_mapper = vtk.vtkPolyDataMapper()
        axes_mapper.SetInputData(axes_pd)
        axes_mapper.SetColorModeToDirectScalars()
        axes_actor = vtk.vtkActor()
        axes_actor.SetMapper(axes_mapper)
        axes_actor.GetProperty().SetLineWidth(2.5)  # Thicker than grid
        axes_actor.GetProperty().LightingOff()
        axes_actor.SetPickable(0)
        axes_actor.UseBoundsOff()   # exclude from ResetCamera bounds

        return grid_actor, axes_actor

    def _set_camera_view(self, position, view_up):
        """Sets the camera to look at the focal point from the given relative direction."""
        import math
        camera = self.renderer.GetActiveCamera()

        # Normalize the position direction vector so distance is consistent
        pos = np.array(position, dtype=float)
        mag = np.linalg.norm(pos)
        if mag > 1e-10:
            pos = pos / mag

        # Compute focal point and distance from visible geometry bounds
        bounds = self.renderer.ComputeVisiblePropBounds()
        if bounds and len(bounds) == 6 and bounds[0] <= bounds[1]:
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            dz = bounds[5] - bounds[4]
            max_dim = max(dx, dy, dz, 1e-3)
            distance = max_dim * 2.5
            focal_point = (
                (bounds[0] + bounds[1]) / 2.0,
                (bounds[2] + bounds[3]) / 2.0,
                (bounds[4] + bounds[5]) / 2.0,
            )
        else:
            distance = camera.GetDistance()
            focal_point = camera.GetFocalPoint()

        camera.SetFocalPoint(*focal_point)
        camera.SetPosition(
            focal_point[0] + pos[0] * distance,
            focal_point[1] + pos[1] * distance,
            focal_point[2] + pos[2] * distance,
        )
        camera.SetViewUp(*view_up)
        self.renderer.ResetCamera()   # fit geometry to view for every standard view
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()
        # Immediately sync the NavCube to the new orientation
        if hasattr(self, '_nav_cube'):
            self._nav_cube.update_rotation(camera)

    def _roll_camera(self, angle):
        """Rolls the camera around its viewing axis by the given angle in degrees."""
        camera = self.renderer.GetActiveCamera()
        camera.Roll(angle)
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()
        if hasattr(self, '_nav_cube'):
            self._nav_cube.update_rotation(camera)

    def _update_scalar_bar(self, title, min_val, max_val, lut=None):
        """Update and show the scalar bar."""
        if not title:
            self.scalar_bar.VisibilityOff()
            return

        self.scalar_bar.SetTitle(title)
        self.scalar_bar.SetNumberOfLabels(5)

        if lut:
            self.scalar_bar.SetLookupTable(lut)
        elif self.current_actor and self.current_actor.GetMapper():
            self.scalar_bar.SetLookupTable(self.current_actor.GetMapper().GetLookupTable())

        self.scalar_bar.VisibilityOn()

    # ──────────────────────────────────────────────────────────────────────────
    # SKETCH RENDER
    # ──────────────────────────────────────────────────────────────────────────

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
                    except Exception:
                        pass

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
                    except Exception:
                        pass

                if hasattr(shape, 'Wires'):
                    try:
                        wires = shape.Wires()
                        for wire in wires:
                            if hasattr(wire, 'Edges'):
                                edges.extend(wire.Edges())
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
                        sampler = GCPnts_UniformAbscissa(curve, 30)

                        if sampler.NbPoints() > 0:
                            for i in range(1, sampler.NbPoints() + 1):
                                p = curve.Value(sampler.Parameter(i))
                                pts_extracted.append((p.X(), p.Y(), p.Z()))
                    except Exception:
                        pass

                # Method 2: CadQuery edge discretization
                if not pts_extracted and hasattr(edge, 'discretize'):
                    try:
                        pts_extracted = edge.discretize(30)
                    except Exception:
                        pass

                # Method 3: Simple start/end for straight lines
                if not pts_extracted and hasattr(edge, 'startPoint') and hasattr(edge, 'endPoint'):
                    sp = edge.startPoint()
                    ep = edge.endPoint()
                    if abs(sp.x - ep.x) > 1e-6 or abs(sp.y - ep.y) > 1e-6 or abs(sp.z - ep.z) > 1e-6:
                        pts_extracted = [(sp.x, sp.y, sp.z), (ep.x, ep.y, ep.z)]

                if pts_extracted and len(pts_extracted) >= 2:
                    start_id = point_id
                    for pt in pts_extracted:
                        if hasattr(pt, 'x'):
                            points.InsertNextPoint(pt.x, pt.y, pt.z if hasattr(pt, 'z') else 0)
                        else:
                            points.InsertNextPoint(pt[0], pt[1], pt[2] if len(pt) > 2 else 0)
                        point_id += 1

                    for i in range(start_id, point_id - 1):
                        line = vtk.vtkLine()
                        line.GetPointIds().SetId(0, i)
                        line.GetPointIds().SetId(1, i + 1)
                        lines.InsertNextCell(line)

            except Exception:
                continue

        if point_id == 0:
            return

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.9, 0.9)  # Cyan
        actor.GetProperty().SetLineWidth(2.0)

        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 0, 100)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)

        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    # ──────────────────────────────────────────────────────────────────────────
    # SHAPE RENDER (with per-face picking support)
    # ──────────────────────────────────────────────────────────────────────────

    def render_shape(self, shape):
        """
        Accepts a CadQuery Workplane or Shape, tessellates it, and renders it.
        Also builds the per-face mapping required for interactive picking.
        """
        self.scalar_bar.VisibilityOff()
        if shape is None:
            return

        self.clear()

        # Reset picking data from previous shape
        self._face_map = {}
        self._all_occ_faces = []
        self._face_polydata_list = []

        topo_shape = shape

        if hasattr(shape, 'toCompound'):
            try:
                topo_shape = shape.toCompound()
            except Exception:
                return

        try:
            if hasattr(topo_shape, 'val'):
                topo_shape = topo_shape.val()
        except Exception:
            pass

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

        if not hasattr(topo_shape, 'tessellate'):
            self.render_sketch(shape)
            return

        # ── Build per-face tessellation for picking ──
        # Try to get OCC faces list for picking
        occ_faces = []
        try:
            if hasattr(topo_shape, 'Faces'):
                occ_faces = topo_shape.Faces()
        except Exception:
            pass

        self._all_occ_faces = occ_faces

        # Build combined poly data for render + per-face polydata for picking
        combined_points = vtk.vtkPoints()
        combined_polys = vtk.vtkCellArray()
        global_cell_idx = 0
        self._face_polydata_list = []

        # We'll try per-face tessellation first (enables picking), fall back to
        # whole-shape tessellation if per-face isn't available.
        use_per_face = len(occ_faces) > 0

        if use_per_face:
            for face_idx, occ_face in enumerate(occ_faces):
                face_pd = vtk.vtkPolyData()
                face_points = vtk.vtkPoints()
                face_polys = vtk.vtkCellArray()

                try:
                    tri = occ_face.tessellate(tolerance=0.1, angularTolerance=0.2)
                    if isinstance(tri, dict):
                        verts = tri.get('vertices') or tri.get('verts') or []
                        triangles = tri.get('triangles') or tri.get('faces') or []
                    else:
                        verts, triangles = tri[0], tri[1]

                    if not verts:
                        self._face_polydata_list.append(None)
                        continue

                    pt_offset = combined_points.GetNumberOfPoints()
                    local_pt_offset = face_points.GetNumberOfPoints()

                    for v in verts:
                        combined_points.InsertNextPoint(v.x, v.y, v.z)
                        face_points.InsertNextPoint(v.x, v.y, v.z)

                    for t in triangles:
                        # Combined mesh
                        combined_polys.InsertNextCell(3)
                        combined_polys.InsertCellPoint(t[0] + pt_offset)
                        combined_polys.InsertCellPoint(t[1] + pt_offset)
                        combined_polys.InsertCellPoint(t[2] + pt_offset)
                        self._face_map[global_cell_idx] = face_idx
                        global_cell_idx += 1

                        # Per-face mesh
                        face_polys.InsertNextCell(3)
                        face_polys.InsertCellPoint(t[0] + local_pt_offset)
                        face_polys.InsertCellPoint(t[1] + local_pt_offset)
                        face_polys.InsertCellPoint(t[2] + local_pt_offset)

                    face_pd.SetPoints(face_points)
                    face_pd.SetPolys(face_polys)
                    self._face_polydata_list.append(face_pd)

                except Exception:
                    self._face_polydata_list.append(None)
                    continue

            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(combined_points)
            poly_data.SetPolys(combined_polys)

        else:
            # Fallback: whole-shape tessellation (no per-face picking)
            try:
                triangulation = topo_shape.tessellate(tolerance=0.1, angularTolerance=0.2)
                if isinstance(triangulation, dict):
                    verts = triangulation.get('vertices') or triangulation.get('verts')
                    triangles = triangulation.get('triangles') or triangulation.get('faces')
                else:
                    verts, triangles = triangulation[0], triangulation[1]
            except Exception:
                self.render_sketch(shape)
                return

            if not verts or len(verts) == 0:
                self.render_sketch(shape)
                return

            points = vtk.vtkPoints()
            polys = vtk.vtkCellArray()

            for v in verts:
                points.InsertNextPoint(v.x, v.y, v.z)
            for t in triangles:
                polys.InsertNextCell(3)
                polys.InsertCellPoint(t[0])
                polys.InsertCellPoint(t[1])
                polys.InsertCellPoint(t[2])

            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            poly_data.SetPolys(polys)

        if combined_points.GetNumberOfPoints() == 0 and not use_per_face:
            self.render_sketch(shape)
            return

        # Calculate Normals for smooth shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly_data)
        normals.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Styling (Metallic Blue/Grey)
        actor.GetProperty().SetColor(0.7, 0.75, 0.8)
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(20)

        self.renderer.AddActor(actor)
        self.current_actor = actor

        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    # ──────────────────────────────────────────────────────────────────────────
    # BC OVERLAYS
    # ──────────────────────────────────────────────────────────────────────────

    def set_bc_overlay_data(self, constraint_faces=None, load_faces=None, load_vectors=None):
        """
        Cache BC overlay data so it can be re-applied after render_simulation().
        Pass None to all arguments to clear the cached data.
        """
        has_data = bool(
            (constraint_faces and any(f is not None for f in constraint_faces)) or
            (load_faces and any(f is not None for f in load_faces)) or
            (load_vectors and len(load_vectors) > 0)
        )
        if has_data:
            self._cached_bc_data = (
                list(constraint_faces) if constraint_faces else [],
                list(load_faces) if load_faces else [],
                list(load_vectors) if load_vectors else [],
            )
        else:
            self._cached_bc_data = None

    def render_bc_overlays(self, constraint_faces=None, load_faces=None, load_vectors=None):
        """
        Highlight boundary condition faces in the viewer:
          - constraint_faces: list of OCC face objects → red highlight
          - load_faces: list of OCC face objects → yellow highlight
          - load_vectors: list of (centroid [x,y,z], vector [fx,fy,fz]) → force arrows
        Always replaces previously drawn BC overlays (no stale arrows/highlights).
        """
        # ── Remove all previous BC overlay actors ──────────────────────────────
        for actor in list(self._bc_overlay_actors):
            self.renderer.RemoveActor(actor)
            # Also pull out of the generic actors list if it ended up there
            if actor in self.actors:
                self.actors.remove(actor)
        self._bc_overlay_actors = []

        def _triangulate_face(occ_face):
            try:
                tri = occ_face.tessellate(tolerance=0.15, angularTolerance=0.3)
                if isinstance(tri, dict):
                    verts = tri.get('vertices') or tri.get('verts') or []
                    tris = tri.get('triangles') or tri.get('faces') or []
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
                boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
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
                order = np.argsort(np.asarray(tri_areas, dtype=float))[::-1]
                selected = []
                min_spacing = 0.0
                try:
                    bb = occ_face.BoundingBox()
                    diag = ((bb.xmax - bb.xmin) ** 2 + (bb.ymax - bb.ymin) ** 2 + (bb.zmax - bb.zmin) ** 2) ** 0.5
                    min_spacing = max(0.25, 0.12 * diag)
                except Exception:
                    min_spacing = 0.25
                for idx in order:
                    p = np.asarray(tri_centroids[int(idx)], dtype=float)
                    if all(np.linalg.norm(p - q) >= min_spacing for q in selected):
                        selected.append(p)
                    if len(selected) >= max_points:
                        break
                return [p.tolist() for p in selected]
            except Exception:
                return []

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
                actor._bc_overlay = True
                self.renderer.AddActor(actor)
                self._bc_overlay_actors.append(actor)
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
            actor._bc_overlay = True
            self.renderer.AddActor(actor)
            self._bc_overlay_actors.append(actor)

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
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(*color)
            actor._bc_overlay = True
            self.renderer.AddActor(actor)
            self._bc_overlay_actors.append(actor)

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
            norm_vec = vector / length
            transform = vtk.vtkTransform()
            transform.Translate(start[0], start[1], start[2])
            default_dir = _np.array([1.0, 0.0, 0.0])
            axis = _np.cross(default_dir, norm_vec)
            angle_deg = _np.degrees(_np.arccos(_np.clip(_np.dot(default_dir, norm_vec), -1.0, 1.0)))
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
            actor._bc_overlay = True
            self.renderer.AddActor(actor)
            self._bc_overlay_actors.append(actor)

        def _overlay_scale():
            try:
                _b = self.renderer.GetBounds()
                _diag = ((_b[1]-_b[0])**2 + (_b[3]-_b[2])**2 + (_b[5]-_b[4])**2) ** 0.5
                return max(2.5, 0.04 * _diag)
            except Exception:
                return 5.0

        def _add_constraint_direction_overlay(occ_face, viz_meta, color):
            sample_points = _sample_face_points(occ_face, max_points=5)
            if not sample_points:
                return

            arrow_len = _overlay_scale()
            fixed_dofs = list(viz_meta.get('fixed_dofs') or [])
            disp_vals = viz_meta.get('displacement')

            if disp_vals is not None:
                disp_vec = np.array(disp_vals, dtype=float)
                if np.linalg.norm(disp_vec) > 1e-12:
                    vis_vec = disp_vec / np.linalg.norm(disp_vec) * arrow_len
                    for point in sample_points:
                        start = np.array(point, dtype=float) - 0.5 * vis_vec
                        _add_bc_arrow(start.tolist(), vis_vec.tolist(), color)
                return

            axis_dirs = {
                0: np.array([1.0, 0.0, 0.0]),
                1: np.array([0.0, 1.0, 0.0]),
                2: np.array([0.0, 0.0, 1.0]),
            }
            if len(fixed_dofs) == 1 and fixed_dofs[0] in axis_dirs:
                direction = axis_dirs[fixed_dofs[0]]
                for point in sample_points:
                    start = np.array(point, dtype=float) - 0.5 * arrow_len * direction
                    _add_bc_arrow(start.tolist(), (arrow_len * direction).tolist(), color)

        if constraint_faces:
            for item in constraint_faces:
                if item is None:
                    continue
                if isinstance(item, dict) and item.get('pos') is not None:
                    viz_meta = item.get('viz', {}) or {}
                    hex_col = viz_meta.get('color', '#2979FF')
                    try:
                        r = int(hex_col[1:3], 16) / 255.0
                        g = int(hex_col[3:5], 16) / 255.0
                        b = int(hex_col[5:7], 16) / 255.0
                    except Exception:
                        r, g, b = 0.16, 0.47, 1.0
                    fixed_dofs = viz_meta.get('fixed_dofs') or [0, 1, 2]
                    points = item.get('points') or [item.get('pos')]
                    for point in points:
                        before_count = len(self.actors)
                        self._add_constraint_glyph(
                            point, fixed_dofs=fixed_dofs, color=(r, g, b),
                            size=max(1.2, 0.65 * _overlay_scale()),
                        )
                        for actor in self.actors[before_count:]:
                            actor._bc_overlay = True
                            self._bc_overlay_actors.append(actor)
                    continue
                # item can be a bare OCC face (legacy) or a dict with 'face' + 'viz' keys
                if isinstance(item, dict):
                    occ_face = item.get('face')
                    viz_meta = item.get('viz', {})
                    # FIX #V4: Per constraint-type colour from viz metadata.
                    hex_col = viz_meta.get('color', '#FF3333') if viz_meta else '#FF3333'
                    # Convert hex #RRGGBB → (r,g,b) floats 0-1
                    try:
                        r = int(hex_col[1:3], 16) / 255.0
                        g = int(hex_col[3:5], 16) / 255.0
                        b = int(hex_col[5:7], 16) / 255.0
                    except Exception:
                        r, g, b = 1.0, 0.2, 0.2
                    _label = viz_meta.get('constraint_type', 'Fixed') if viz_meta else 'Fixed'
                    _ = _label  # may be used for a 3D text label in the future
                else:
                    occ_face = item
                    r, g, b = 1.0, 0.2, 0.2  # legacy: red
                if occ_face is not None:
                    _add_face_outline(occ_face, color=(r, g, b), line_width=3.0)
                    if isinstance(item, dict):
                        _add_constraint_direction_overlay(occ_face, viz_meta, (r, g, b))

        if load_faces:
            for item in load_faces:
                if item is None:
                    continue
                occ_face = item.get('face') if isinstance(item, dict) else item
                if occ_face is not None:
                    _add_face_outline(occ_face, color=(1.0, 0.85, 0.0), line_width=2.5)
                    _add_face_overlay(occ_face, color=(1.0, 0.85, 0.0), opacity=0.08)

        if load_vectors:
            import math as _math
            for entry in load_vectors:
                if isinstance(entry, dict):
                    centroid  = entry['centroid']
                    occ_face   = entry.get('face')
                    vec       = np.array(entry['vector'], dtype=float)
                    force_mag = float(entry.get('magnitude_N', np.linalg.norm(vec)))
                    # Per-entry color: hex string or default yellow
                    _hex = entry.get('color', '#FFD900')
                    try:
                        _hx = _hex.lstrip('#')
                        arrow_color = tuple(int(_hx[i:i+2], 16)/255.0 for i in (0, 2, 4))
                    except Exception:
                        arrow_color = (1.0, 0.85, 0.0)
                else:
                    # legacy (centroid, vec) tuple
                    centroid, vec = entry
                    occ_face = None
                    force_mag = float(np.linalg.norm(vec))
                    arrow_color = (1.0, 0.85, 0.0)
                # Log-scale arrow length so magnitude is visually conveyed.
                try:
                    _b = self.renderer.GetBounds()
                    _diag = ((_b[1]-_b[0])**2 + (_b[3]-_b[2])**2 + (_b[5]-_b[4])**2) ** 0.5
                    base_len = max(5.0, 0.08 * _diag)
                except Exception:
                    base_len = 5.0
                log_s = float(np.clip(_math.log10(force_mag + 1) / 6.0, 0.15, 1.0))
                arrow_len = base_len * (0.5 + log_s)
                unit_vec  = vec / (np.linalg.norm(vec) + 1e-12)
                vis_vec   = unit_vec * arrow_len
                sample_points = _sample_face_points(occ_face, max_points=5) if occ_face is not None else []
                if not sample_points and isinstance(entry, dict):
                    sample_points = entry.get('points') or []
                if sample_points:
                    scaled_vec = unit_vec * max(0.45 * arrow_len, 0.8 * _overlay_scale())
                    for point in sample_points:
                        arrow_start = np.array(point, dtype=float) - scaled_vec
                        _add_bc_arrow(arrow_start.tolist(), scaled_vec, color=arrow_color)
                else:
                    arrow_start = np.array(centroid, dtype=float) - vis_vec
                    _add_bc_arrow(arrow_start.tolist(), vis_vec, color=arrow_color)

        self.vtkWidget.GetRenderWindow().Render()

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

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

    def _add_constraint_glyph(self, pos, fixed_dofs=None, color=(0.16, 0.47, 1.0), size=2.5):
        """Industry-standard BC glyph.

        Renders different shapes per restraint type:
            * Fixed (3 DOF)  → three orthogonal small cones converging on the
              face point (clamp / ground-anchor convention).
            * Roller / Symmetry (1 DOF) → cylinder lying perpendicular to the
              restrained axis (the "free-to-roll" convention).
            * Pinned (3 DOF for solids) → flat disc.

        Falls back to a small sphere if the DOF pattern is unrecognised.
        """
        fixed_dofs = list(fixed_dofs or [])
        # Axis basis vectors keyed by DOF index (0=X, 1=Y, 2=Z).
        axis_vec = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0)}

        def _make_actor(source, transform=None):
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if transform is not None:
                actor.SetUserTransform(transform)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetAmbient(0.4)
            actor.GetProperty().SetDiffuse(0.6)
            self.renderer.AddActor(actor)
            self.actors.append(actor)
            return actor

        if len(fixed_dofs) >= 3 or not fixed_dofs:
            # Fully restrained — three short cones meeting at the point, plus
            # a tiny ground square to make it look like a clamp.
            for dof in (0, 1, 2):
                dx, dy, dz = axis_vec[dof]
                cone = vtk.vtkConeSource()
                cone.SetHeight(size)
                cone.SetRadius(size * 0.45)
                cone.SetResolution(20)
                t = vtk.vtkTransform()
                t.Translate(pos[0] + dx * size * 0.6,
                            pos[1] + dy * size * 0.6,
                            pos[2] + dz * size * 0.6)
                # vtkConeSource default points along +X; rotate to align with axis.
                if dof == 1:
                    t.RotateZ(90)
                elif dof == 2:
                    t.RotateY(-90)
                # Make the cone tip point inward toward `pos`.
                t.RotateZ(180)
                _make_actor(cone, t)
            return

        if len(fixed_dofs) == 1:
            dof = int(fixed_dofs[0])
            # Cylinder lying along an axis perpendicular to the restrained one
            # (a roller's axis is *not* the restrained direction).  Use the
            # next axis cyclically so each roller looks distinct.
            roller_axis = (dof + 1) % 3
            cyl = vtk.vtkCylinderSource()
            cyl.SetRadius(size * 0.5)
            cyl.SetHeight(size * 2.5)
            cyl.SetResolution(24)
            t = vtk.vtkTransform()
            t.Translate(pos[0], pos[1], pos[2])
            # vtkCylinderSource is +Y aligned by default.
            if roller_axis == 0:
                t.RotateZ(-90)
            elif roller_axis == 2:
                t.RotateX(90)
            _make_actor(cyl, t)
            return

        # Generic 2-DOF fall-through: small sphere.
        sph = vtk.vtkSphereSource()
        sph.SetCenter(pos[0], pos[1], pos[2])
        sph.SetRadius(size * 0.6)
        sph.SetThetaResolution(24)
        sph.SetPhiResolution(16)
        _make_actor(sph)

    def _add_force_label(self, pos, magnitude_text, color=(1.0, 0.85, 0.2)):
        """Add a billboard text label at a 3D point (used for force magnitudes)."""
        try:
            actor = vtk.vtkBillboardTextActor3D()
            actor.SetPosition(pos[0], pos[1], pos[2])
            actor.SetInput(str(magnitude_text))
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

    # ──────────────────────────────────────────────────────────────────────────
    # SIMULATION RENDER
    # ──────────────────────────────────────────────────────────────────────────

    def render_simulation(self, data):
        """
        Render simulation results (Mesh or FEA Result).
        """
        if data is None:
            return

        # Auto-start crash animation playback when solver result with frames arrives
        if (isinstance(data, dict) and
                data.get('type') == 'crash' and
                data.get('frames')):
            self.start_crash_playback(data)
            return

        # 1. Clean up previous render
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None

        # --- RECOVERED SHAPE VIZ (Triangulated Surface) ---
        if isinstance(data, dict) and data.get('visualization_mode') == 'Recovered Shape':
            rec = data.get('recovered_shape')
            if rec and 'vertices' in rec and 'faces' in rec:
                verts = rec['vertices']
                faces = rec['faces']
                
                poly_data = vtk.vtkPolyData()
                points = vtk.vtkPoints()
                for v in verts:
                    points.InsertNextPoint(v[0], v[1], v[2])
                poly_data.SetPoints(points)
                
                cells = vtk.vtkCellArray()
                for f in faces:
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, f[0])
                    triangle.GetPointIds().SetId(1, f[1])
                    triangle.GetPointIds().SetId(2, f[2])
                    cells.InsertNextCell(triangle)
                poly_data.SetPolys(cells)
                
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly_data)
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                # Soft premium gray
                actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                actor.GetProperty().SetOpacity(1.0)
                
                self.renderer.AddActor(actor)
                self.current_actor = actor
                
                # Update scalar bar to be empty for geometry view
                self._update_scalar_bar("", 0, 1, None)
                
                self.vtkWidget.GetRenderWindow().Render()
                return


        # Check if it's a Mesh object or Result dict
        mesh = None
        displacement = None
        density = None
        stress = None
        visualization_mode = 'Von Mises Stress'
        density_cutoff = 0.5
        locked_scalar_range = None   # (lo, hi) supplied by crash playback for stable colormap
        max_stress_gauss    = 0.0    # P0 Gauss-point peak (populated by SolverNode)
        _def_scale          = 1.0    # visualisation deformation scale factor

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
            if '_scalar_range' in data:
                locked_scalar_range = data['_scalar_range']
            if 'max_stress_gauss' in data:
                max_stress_gauss = float(data['max_stress_gauss'])
            if 'deformation_scale' in data:
                _def_scale = float(data['deformation_scale'])

        if mesh is None:
            return

        # 2. Create VTK Unstructured Grid
        points = vtk.vtkPoints()
        grid = vtk.vtkUnstructuredGrid()

        pts = mesh.p
        n_points = pts.shape[1]

        # Apply displacement if available (scaled for visualisation)
        if displacement is not None:
            if len(displacement) == 3 * n_points:
                disp_3n = displacement.reshape((3, n_points), order='F')
                pts = pts + disp_3n * _def_scale

        for i in range(n_points):
            points.InsertNextPoint(pts[0, i], pts[1, i], pts[2, i])

        grid.SetPoints(points)

        tets = mesh.t
        n_tets = tets.shape[1]

        # Add Density Data if available
        if density is not None:
            density_array = vtk.vtkFloatArray()
            density_array.SetName("Density")
            for d in density:
                density_array.InsertNextValue(float(d))
            grid.GetCellData().SetScalars(density_array)

        if stress is not None:
            s_array = vtk.vtkFloatArray()
            s_array.SetName("VonMises")
            for s in stress:
                s_array.InsertNextValue(float(s))
            grid.GetPointData().AddArray(s_array)
            if visualization_mode in ('Von Mises Stress', 'Plastic Strain', 'Failed Elements'):
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
            cutoff = float(np.clip(density_cutoff, 0.05, 0.95))
            lower = max(0.01, cutoff)
            upper = 1.1

            threshold = vtk.vtkThreshold()
            threshold.SetInputData(grid)
            threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Density")
            threshold.SetLowerThreshold(lower)
            threshold.SetUpperThreshold(upper)
            threshold.SetThresholdFunction(getattr(vtk.vtkThreshold, 'THRESHOLD_BETWEEN', 0))

            try:
                threshold.SetPassPointArrays(True)
            except AttributeError:
                pass

            threshold.Update()

            threshold_output = threshold.GetOutput()

            if threshold_output.GetNumberOfCells() == 0 and len(density) > 0:
                relaxed = float(max(0.01, np.percentile(density, 10)))
                threshold.SetLowerThreshold(relaxed)
                threshold.SetUpperThreshold(upper)
                threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
                threshold.Update()
                threshold_output = threshold.GetOutput()

            mapper.SetInputData(threshold_output)
            mapper.SetScalarRange(0, 1)
        else:
            surface = vtk.vtkDataSetSurfaceFilter()
            surface.SetInputData(grid)
            surface.Update()
            mapper.SetInputData(surface.GetOutput())

            dataset = surface.GetOutput()

            if dataset.GetPointData().GetScalars() is not None:
                scalars = dataset.GetPointData().GetScalars()
                if locked_scalar_range is not None:
                    min_val, max_val = locked_scalar_range
                else:
                    min_val, max_val = scalars.GetRange()

                if max_val - min_val < 1e-10:
                    max_val = min_val + 1.0

                mapper.SetScalarModeToUsePointData()
                mapper.SelectColorArray(scalars.GetName())
                mapper.SetScalarRange(min_val, max_val)

                lut = vtk.vtkLookupTable()
                lut.SetHueRange(0.667, 0.0)  # Blue to Red
                lut.Build()
                mapper.SetLookupTable(lut)

                scalar_name = scalars.GetName()
                if scalar_name == "VonMises":
                    if visualization_mode == 'Plastic Strain':
                        self._update_scalar_bar("Equivalent Plastic Strain", min_val, max_val, lut)
                    elif visualization_mode == 'Failed Elements':
                        self._update_scalar_bar("Element Failure (0=intact, 1=failed)", min_val, max_val, lut)
                    else:
                        self._update_scalar_bar("Von Mises Stress (MPa)", min_val, max_val, lut)
                elif scalar_name == "Displacement":
                    _disp_title = (
                        f"Displacement (mm)  [scale: {_def_scale:.0f}\u00d7]"
                        if _def_scale != 1.0 else "Displacement (mm)"
                    )
                    self._update_scalar_bar(_disp_title, min_val, max_val, lut)

            elif stress is not None and len(stress) > 0:
                if locked_scalar_range is not None:
                    min_s, max_s = locked_scalar_range
                else:
                    min_s, max_s = float(np.min(stress)), float(np.max(stress))

                if max_s - min_s < 1e-10:
                    max_s = min_s + 1.0

                mapper.SetScalarRange(min_s, max_s)
                mapper.SetScalarModeToUsePointData()

                lut = vtk.vtkLookupTable()
                lut.SetHueRange(0.667, 0.0)
                lut.Build()
                mapper.SetLookupTable(lut)

                self._update_scalar_bar("Von Mises Stress (MPa)", min_s, max_s, lut)

        if density is not None and stress is not None and visualization_mode == 'Von Mises Stress':
            mapper_input = mapper.GetInput()
            if mapper_input is not None and mapper_input.GetPointData() is not None:
                mapper_input.GetPointData().SetActiveScalars("VonMises")
            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray("VonMises")

            # FIX: Skip void elements when calculating stress range for colorbar
            if hasattr(mesh, 't') and mesh.t.shape[1] == len(density) and len(stress) == mesh.p.shape[1]:
                # Nodes connected to solid elements
                solid_elems = np.where(density >= float(np.clip(density_cutoff, 0.05, 0.95)))[0]
                if len(solid_elems) > 0:
                    solid_nodes = np.unique(mesh.t[:, solid_elems])
                    valid_stress = stress[solid_nodes]
                    min_s, max_s = float(np.min(valid_stress)), float(np.max(valid_stress))
                else:
                    min_s, max_s = float(np.min(stress)), float(np.max(stress))
            else:
                min_s, max_s = float(np.min(stress)), float(np.max(stress))

            if max_s - min_s < 1e-10:
                max_s = min_s + 1.0

            mapper.SetScalarRange(min_s, max_s)

            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)
            lut.Build()
            mapper.SetLookupTable(lut)
            self._update_scalar_bar("Von Mises Stress (MPa)", min_s, max_s, lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        if density is not None and visualization_mode == 'Density':
            actor.GetProperty().SetColor(0.9, 0.7, 0.1)
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()
        elif stress is not None or displacement is not None:
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()
        else:
            actor.GetProperty().SetColor(0.8, 0.4, 0.4)
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()

        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)

        # 4. Debug Visualization (Loads/Constraints after solve)
        # If exact BC overlays are cached from the graph, prefer those over the
        # approximate centroid/bbox debug markers to avoid duplicate clutter.
        _has_exact_bc_overlay = self._cached_bc_data is not None
        if isinstance(data, dict) and not _has_exact_bc_overlay:
            try:
                _b = self.renderer.GetBounds()
                _diag = ((_b[1] - _b[0]) ** 2 + (_b[3] - _b[2]) ** 2 + (_b[5] - _b[4]) ** 2) ** 0.5
                base_len = max(5.0, 0.08 * _diag)
                glyph_size = max(1.5, 0.025 * _diag)
            except Exception:
                base_len, glyph_size = 5.0, 2.0

            def _hex_to_rgb(color, fallback):
                if isinstance(color, str) and color.startswith('#'):
                    c = color.lstrip('#')
                    return tuple(int(c[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                return fallback

            if 'debug_loads' in data and data['debug_loads']:
                for load in data['debug_loads']:
                    if 'vector' not in load:
                        continue
                    raw_vec = np.array(load['vector'], dtype=float)
                    raw_mag = float(np.linalg.norm(raw_vec))
                    if raw_mag < 1e-9:
                        continue
                    norm_dir = raw_vec / raw_mag

                    center = load.get('start') or load.get('pos')
                    if not center and 'bbox' in load:
                        bb = load['bbox']
                        center = [(bb.xmin + bb.xmax) / 2,
                                  (bb.ymin + bb.ymax) / 2,
                                  (bb.zmin + bb.zmax) / 2]
                    if not center:
                        continue

                    rel_mag = load.get('relative_mag', 1.0)
                    arrow_len = base_len * max(0.2, min(1.0, rel_mag)) * 1.5
                    c_rgb = _hex_to_rgb(load.get('color', '#ffff00'), (1.0, 0.85, 0.2))
                    self._add_arrow(center, norm_dir, color=c_rgb, scale=arrow_len / 5.0)

                    # Magnitude label at the arrow tip (only for loads that
                    # actually carry a magnitude; relative_mag * raw_mag
                    # recovers the absolute Newton value used by the solver).
                    label_pos = [center[i] + norm_dir[i] * arrow_len for i in range(3)]
                    label = self._format_force_magnitude(raw_mag)
                    if label:
                        self._add_force_label(label_pos, label, color=c_rgb)

            if 'debug_constraints' in data and data['debug_constraints']:
                for const in data['debug_constraints']:
                    center = const.get('pos')
                    if not center:
                        continue
                    c_rgb = _hex_to_rgb(const.get('color', '#2979FF'), (0.16, 0.47, 1.0))
                    fixed_dofs = const.get('fixed_dofs')
                    self._add_constraint_glyph(
                        center, fixed_dofs=fixed_dofs, color=c_rgb, size=glyph_size
                    )

        # 5. Re-apply cached BC face overlays on top of the simulation result
        #    so they remain visible after FEA/TopOpt solve.
        #    Skip only for TopOpt iteration previews (keep frame rate high).
        #    Crash frames DO replay overlays so supports/impact face stay visible.
        skip_bc_replay = (
            isinstance(data, dict)
            and data.get('type') == 'topopt'
            and bool(data.get('_preview', False))
        )
        if not skip_bc_replay and self._cached_bc_data is not None:
            c_faces, l_faces, l_vecs = self._cached_bc_data
            self.render_bc_overlays(
                constraint_faces=c_faces or None,
                load_faces=l_faces or None,
                load_vectors=l_vecs or None,
            )

        # Reset camera only when explicitly requested (first crash frame) or for
        # non-animation renders.  Skipping ResetCamera on subsequent animation
        # frames keeps the viewpoint fixed so mesh deformation is clearly visible
        # rather than being hidden by continuous re-centring.
        should_reset = True
        if isinstance(data, dict) and data.get('type') == 'crash_frame':
            should_reset = bool(data.get('_reset_camera', False))
        if should_reset:
            self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
