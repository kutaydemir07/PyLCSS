# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Modern graphics painter for NodeGraphQt node items across PyLCSS workbenches."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


def _get_node_category_badge(node_item) -> tuple[str, QtGui.QColor, QtGui.QColor]:
    """Return category title, header start color, and header end color based on node type and role border color."""
    node_type = str(getattr(node_item, "type_", "") or "").lower()
    node_name = str(getattr(node_item, "name", "") or "").lower()

    # Category matching badge text
    badge = ""
    if "input" in node_type or "input" in node_name:
        badge = "INPUT"
    elif "output" in node_type or "output" in node_name:
        badge = "OUTPUT"
    elif "freecad" in node_type or "cad" in node_type or "part" in node_name:
        badge = "CAD"
    elif "mesh" in node_type or "fea" in node_type or "boundary" in node_type:
        badge = "FEA"
    elif "crash" in node_type or "impact" in node_type or "radioss" in node_type:
        badge = "IMPACT"
    elif "topopt" in node_type or "topology" in node_type or "opt" in node_type:
        badge = "OPT"
    elif "custom_block" in node_type or "code" in node_type or "function" in node_name:
        badge = "MATH"

    # Default category fallback gradients
    default_colors = {
        "INPUT": (QtGui.QColor(55, 177, 224, 240), QtGui.QColor(30, 115, 150, 240)),
        "OUTPUT": (QtGui.QColor(63, 192, 147, 240), QtGui.QColor(32, 125, 95, 240)),
        "CAD": (QtGui.QColor(14, 116, 144, 240), QtGui.QColor(8, 75, 100, 240)),
        "FEA": (QtGui.QColor(124, 58, 237, 240), QtGui.QColor(88, 28, 185, 240)),
        "IMPACT": (QtGui.QColor(217, 119, 6, 240), QtGui.QColor(160, 80, 2, 240)),
        "OPT": (QtGui.QColor(13, 148, 136, 240), QtGui.QColor(8, 95, 88, 240)),
        "MATH": (QtGui.QColor(148, 107, 220, 240), QtGui.QColor(95, 65, 155, 240)),
    }

    start_col, end_col = default_colors.get(
        badge, (QtGui.QColor(42, 48, 60, 240), QtGui.QColor(30, 35, 45, 240))
    )

    # Derive directly from node border_color if explicitly defined
    if hasattr(node_item, "border_color") and node_item.border_color:
        b_rgb = node_item.border_color[:3]
        if max(b_rgb) > 40:
            start_col = QtGui.QColor(b_rgb[0], b_rgb[1], b_rgb[2], 240)
            end_col = QtGui.QColor(
                int(b_rgb[0] * 0.65), int(b_rgb[1] * 0.65), int(b_rgb[2] * 0.65), 240
            )

    return badge, start_col, end_col



def _is_light_theme(item) -> bool:
    """Return whether the current active theme is Light Mode."""
    try:
        from pylcss.user_interface.common.theme_manager import current_theme

        if current_theme() == "light":
            return True
    except Exception:
        pass
    viewer = item.viewer() if hasattr(item, "viewer") else None
    if viewer and hasattr(viewer, "background_color"):
        bg_col = viewer.background_color()
        if isinstance(bg_col, (list, tuple)) and len(bg_col) >= 3:
            return (bg_col[0] + bg_col[1] + bg_col[2]) / 3.0 > 160
    return False


def paint_modern_node_horizontal(node_item, painter: QtGui.QPainter, option, widget) -> None:
    """High-quality modern painter for NodeGraphQt horizontal nodes.

    Replaces classic flat rectangle with rounded squircle cards, drop shadows,
    gradient accent headers, category badges, and sleek borders.
    """
    painter.save()
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtCore.Qt.NoBrush)

    # Base dimensions
    margin = 1.0
    rect = node_item.boundingRect()
    node_rect = QtCore.QRectF(
        rect.left() + margin,
        rect.top() + margin,
        rect.width() - (margin * 2),
        rect.height() - (margin * 2),
    )

    is_light = _is_light_theme(node_item)

    # Geometry corner radius
    node_type_str = str(getattr(node_item, "type_", "")).lower()
    is_terminal = "input" in node_type_str or "output" in node_type_str
    radius = 14.0 if is_terminal else 10.0

    # 1. Drop Shadow & Ambient Glow
    if node_item.selected:
        glow_color = QtGui.QColor(88, 140, 245, 70)
        for step in (6.0, 4.0, 2.0):
            glow_rect = node_rect.adjusted(-step, -step, step, step)
            glow_path = QtGui.QPainterPath()
            glow_path.addRoundedRect(glow_rect, radius + step, radius + step)
            painter.fillPath(glow_path, glow_color)
    else:
        # Subtle drop shadow
        shadow_rect = node_rect.adjusted(1.0, 2.0, 2.0, 3.0)
        shadow_path = QtGui.QPainterPath()
        shadow_path.addRoundedRect(shadow_rect, radius, radius)
        shadow_col = QtGui.QColor(0, 0, 0, 18) if is_light else QtGui.QColor(0, 0, 0, 45)
        painter.fillPath(shadow_path, shadow_col)

    # 2. Main Node Body Fill
    body_path = QtGui.QPainterPath()
    body_path.addRoundedRect(node_rect, radius, radius)

    if is_light:
        body_bg = QtGui.QColor(255, 255, 255, 255)
        if hasattr(node_item, "_set_text_color"):
            node_item._set_text_color((30, 41, 59, 255))
    else:
        body_bg = QtGui.QColor(28, 35, 48, 250)
        if hasattr(node_item, "_set_text_color"):
            node_item._set_text_color((248, 250, 252, 255))
    painter.fillPath(body_path, body_bg)

    # 3. Header Region & Gradient Banner
    header_height = 28.0
    if hasattr(node_item, "_text_item") and node_item._text_item:
        text_h = node_item._text_item.boundingRect().height()
        if text_h > 0:
            header_height = max(26.0, text_h + 4.0)

    header_rect = QtCore.QRectF(
        node_rect.left(),
        node_rect.top(),
        node_rect.width(),
        header_height,
    )

    badge_text, head_start, head_end = _get_node_category_badge(node_item)
    if is_light:
        head_start = head_start.lighter(145)
        head_end = head_end.lighter(135)

    header_gradient = QtGui.QLinearGradient(
        header_rect.left(), header_rect.top(), header_rect.right(), header_rect.bottom()
    )
    header_gradient.setColorAt(0.0, head_start)
    header_gradient.setColorAt(1.0, head_end)

    # Header clip path (rounded top corners, square bottom)
    header_path = QtGui.QPainterPath()
    header_path.moveTo(node_rect.left() + radius, node_rect.top())
    header_path.lineTo(node_rect.right() - radius, node_rect.top())
    header_path.quadTo(node_rect.right(), node_rect.top(), node_rect.right(), node_rect.top() + radius)
    header_path.lineTo(node_rect.right(), header_rect.bottom())
    header_path.lineTo(node_rect.left(), header_rect.bottom())
    header_path.lineTo(node_rect.left(), node_rect.top() + radius)
    header_path.quadTo(node_rect.left(), node_rect.top(), node_rect.left() + radius, node_rect.top())

    painter.fillPath(header_path, header_gradient)

    # Subtle horizontal line under header
    sep_pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 18) if is_light else QtGui.QColor(255, 255, 255, 35), 1.0)
    painter.setPen(sep_pen)
    painter.drawLine(
        QtCore.QPointF(node_rect.left(), header_rect.bottom()),
        QtCore.QPointF(node_rect.right(), header_rect.bottom()),
    )

    # 4. Header Badge Chip (if present)
    if badge_text:
        badge_font = QtGui.QFont("Segoe UI", 7, QtGui.QFont.Bold)
        fm = QtGui.QFontMetrics(badge_font)
        bw = fm.horizontalAdvance(badge_text) + 10.0
        bh = 14.0
        bx = node_rect.right() - bw - 8.0
        by = node_rect.top() + (header_height - bh) / 2.0

        badge_rect = QtCore.QRectF(bx, by, bw, bh)
        badge_path = QtGui.QPainterPath()
        badge_path.addRoundedRect(badge_rect, 4.0, 4.0)

        if is_light:
            painter.fillPath(badge_path, QtGui.QColor(0, 0, 0, 15))
            painter.setFont(badge_font)
            painter.setPen(QtGui.QColor(30, 41, 59, 220))
        else:
            painter.fillPath(badge_path, QtGui.QColor(255, 255, 255, 30))
            painter.setFont(badge_font)
            painter.setPen(QtGui.QColor(255, 255, 255, 220))

        painter.drawText(badge_rect, QtCore.Qt.AlignCenter, badge_text)

    # 5. Inner Glass Highlight Inset
    glass_pen = QtGui.QPen(
        QtGui.QColor(255, 255, 255, 140) if is_light else QtGui.QColor(255, 255, 255, 30),
        1.0,
    )
    painter.setPen(glass_pen)
    glass_rect = node_rect.adjusted(1.0, 1.0, -1.0, -1.0)
    glass_path = QtGui.QPainterPath()
    glass_path.addRoundedRect(glass_rect, radius - 1.0, radius - 1.0)
    painter.drawPath(glass_path)

    # 6. Outer Border Outline
    if node_item.selected:
        border_width = 2.0
        border_color = QtGui.QColor(90, 160, 255, 255)
    else:
        border_width = 1.0
        if is_light:
            border_color = QtGui.QColor(205, 215, 228, 255)
        else:
            border_color = QtGui.QColor(58, 66, 78, 220)

    border_pen = QtGui.QPen(border_color, border_width)
    if hasattr(node_item, "viewer") and node_item.viewer():
        border_pen.setCosmetic(node_item.viewer().get_zoom() < 0.0)

    painter.setPen(border_pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    painter.drawPath(body_path)

    painter.restore()




def paint_modern_port(port_item, painter: QtGui.QPainter, option, widget) -> None:
    """Modern glowing socket painter for NodeGraphQt port items."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

    w = getattr(port_item, "_width", 10.0) / 1.7
    h = getattr(port_item, "_height", 10.0) / 1.7
    center = port_item.boundingRect().center()
    port_rect = QtCore.QRectF(center.x() - w / 2.0, center.y() - h / 2.0, w, h)

    is_hovered = getattr(port_item, "_hovered", False)
    is_connected = bool(getattr(port_item, "connected_pipes", None))

    port_col = getattr(port_item, "color", (200, 200, 200))
    border_col = getattr(port_item, "border_color", (240, 240, 240))

    base_color = QtGui.QColor(port_col[0], port_col[1], port_col[2], 255)
    border_color = QtGui.QColor(border_col[0], border_col[1], border_col[2], 255)

    # Ambient halo glow when hovered or connected
    if is_hovered:
        halo_col = QtGui.QColor(90, 160, 255, 100)
        halo_rect = port_rect.adjusted(-3.0, -3.0, 3.0, 3.0)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(halo_col)
        painter.drawEllipse(halo_rect)
        border_color = QtGui.QColor(120, 190, 255, 255)
    elif is_connected:
        halo_col = QtGui.QColor(border_color.red(), border_color.green(), border_color.blue(), 60)
        halo_rect = port_rect.adjusted(-2.0, -2.0, 2.0, 2.0)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(halo_col)
        painter.drawEllipse(halo_rect)

    # Socket outer stroke & fill
    pen = QtGui.QPen(border_color, 1.6)
    painter.setPen(pen)
    painter.setBrush(base_color if not is_connected else border_color.darker(110))
    painter.drawEllipse(port_rect)

    # Core inner socket dot
    if is_connected or is_hovered:
        core_w = port_rect.width() * 0.45
        core_h = port_rect.height() * 0.45
        core_rect = QtCore.QRectF(center.x() - core_w / 2.0, center.y() - core_h / 2.0, core_w, core_h)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 240) if is_hovered else base_color)
        painter.drawEllipse(core_rect)

    painter.restore()


def install_modern_port_painter() -> bool:
    """Install the modern painter into NodeGraphQt PortItem."""
    try:
        from NodeGraphQt.qgraphics.port import PortItem
    except (ImportError, AttributeError):
        return False

    original = getattr(PortItem, "paint", None)
    if getattr(original, "_pylcss_modern_port_painter", False):
        return True

    def safe_modern_port_paint(self, painter, option, widget):
        try:
            paint_modern_port(self, painter, option, widget)
        except Exception:
            if original:
                original(self, painter, option, widget)

    safe_modern_port_paint._pylcss_modern_port_painter = True
    PortItem.paint = safe_modern_port_paint
    return True


def paint_modern_pipe(pipe_item, painter: QtGui.QPainter, option, widget) -> None:
    """Modern pipe connection painter with smooth glowing curves and sleek directional arrowheads."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

    path = pipe_item.path()
    if path.isEmpty() or path.length() < 5.0:
        painter.restore()
        return

    # Determine theme environment
    viewer = pipe_item.viewer() if hasattr(pipe_item, "viewer") else None
    is_light = False
    if viewer and hasattr(viewer, "background_color"):
        bg_col = viewer.background_color()
        if isinstance(bg_col, (list, tuple)) and len(bg_col) >= 3:
            is_light = (bg_col[0] + bg_col[1] + bg_col[2]) / 3.0 > 160

    is_selected = getattr(pipe_item, "selected", False) or getattr(pipe_item, "_active", False)
    is_highlighted = getattr(pipe_item, "_highlight", False)
    is_disabled = getattr(pipe_item, "disabled", lambda: False)()

    # Color palette (All Orange, matching original NodeGraph style)
    if is_disabled:
        pipe_color = QtGui.QColor(140, 145, 155, 120)
        pipe_width = 2.0
    elif is_selected:
        pipe_color = QtGui.QColor(255, 200, 50, 255)
        pipe_width = 2.5
    elif is_highlighted:
        pipe_color = QtGui.QColor(255, 175, 40, 255)
        pipe_width = 3.0
    else:
        pipe_color = QtGui.QColor(225, 150, 35, 240)
        pipe_width = 2.0


    # Ambient glow for selected/highlighted pipe
    if (is_selected or is_highlighted) and not is_disabled:
        glow_pen = QtGui.QPen(
            QtGui.QColor(pipe_color.red(), pipe_color.green(), pipe_color.blue(), 70),
            pipe_width + 3.0,
        )
        painter.setPen(glow_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(path)

    # Main pipe curve
    pen = QtGui.QPen(pipe_color, pipe_width)
    if is_disabled:
        pen.setStyle(QtCore.Qt.DashLine)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    painter.drawPath(path)

    painter.restore()



def install_modern_pipe_painter() -> bool:
    """Install the modern painter into NodeGraphQt PipeItem."""
    try:
        from NodeGraphQt.qgraphics.pipe import PipeItem
    except (ImportError, AttributeError):
        return False

    original = getattr(PipeItem, "paint", None)
    if getattr(original, "_pylcss_modern_pipe_painter", False):
        return True

    def safe_modern_pipe_paint(self, painter, option, widget):
        try:
            paint_modern_pipe(self, painter, option, widget)
        except Exception:
            if original:
                original(self, painter, option, widget)

    safe_modern_pipe_paint._pylcss_modern_pipe_painter = True
    PipeItem.paint = safe_modern_pipe_paint
    return True


def install_modern_node_painter() -> bool:
    """Install the modern painter into NodeGraphQt NodeItem, PortItem, and PipeItem."""
    install_modern_port_painter()
    install_modern_pipe_painter()
    try:
        from NodeGraphQt.qgraphics.node_base import NodeItem
    except (ImportError, AttributeError):
        return False

    original = getattr(NodeItem, "_paint_horizontal", None)
    if getattr(original, "_pylcss_modern_painter", False):
        return True

    def safe_modern_paint(self, painter, option, widget):
        try:
            paint_modern_node_horizontal(self, painter, option, widget)
        except Exception:
            if original:
                original(self, painter, option, widget)

    safe_modern_paint._pylcss_modern_painter = True
    NodeItem._paint_horizontal = safe_modern_paint
    return True


