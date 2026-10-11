# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Modern graphics painter for NodeGraphQt node items across PyLCSS workbenches."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


def _get_node_category_badge(node_item) -> tuple[str, QtGui.QColor]:
    """Return category title and calm, disciplined accent color based on node type."""
    node_type = str(getattr(node_item, "type_", "") or "").lower()
    node_name = str(getattr(node_item, "name", "") or "").lower()

    badge = ""
    # Specific engineering disciplines checked first before generic substrings
    if "crash" in node_type or "impact" in node_type or "radioss" in node_type:
        badge = "IMPACT"
    elif "topopt" in node_type or "topology" in node_type or "lattice" in node_type:
        badge = "OPT"
    elif "constraint" in node_type or "pressure" in node_type or "load" in node_type:
        badge = "LOAD"
    elif "mesh" in node_type or "fea" in node_type or "material" in node_type or "component" in node_type or "solver" in node_type:
        badge = "FEA"
    elif "number" in node_type or "variable" in node_type or "select" in node_type or ("input" in node_type or "input" in node_name):
        badge = "INPUT"
    elif "export" in node_type or "output" in node_type or "output" in node_name or "analysis" in node_type or "properties" in node_type:
        badge = "OUTPUT"
    elif "custom_block" in node_type or "code" in node_type or "function" in node_name or "math" in node_type:
        badge = "MATH"
    elif "freecad" in node_type or "cad" in node_type or "part" in node_name or "geometry" in node_type or "box" in node_type or "cylinder" in node_type or "boolean" in node_type or "pattern" in node_type or "transform" in node_type or "assembly" in node_type:
        badge = "CAD"

    # Calm, disciplined engineering accents (harmonized with MDAO palette)
    accent_colors = {
        "INPUT": QtGui.QColor(70, 130, 180),     # Steel blue (matches MDAO input)
        "OUTPUT": QtGui.QColor(76, 140, 100),    # Muted sage (matches MDAO output)
        "CAD": QtGui.QColor(85, 115, 150),      # Slate blue / steel gray
        "FEA": QtGui.QColor(75, 110, 175),      # Muted navy blue
        "LOAD": QtGui.QColor(195, 95, 80),       # Warm terracotta
        "IMPACT": QtGui.QColor(180, 85, 75),     # Muted crimson
        "OPT": QtGui.QColor(210, 153, 34),       # Warm amber (matches MDAO primary #d29922)
        "MATH": QtGui.QColor(130, 105, 160),     # Muted plum (matches MDAO custom block)
    }

    accent = accent_colors.get(badge, QtGui.QColor(100, 116, 139))
    return badge, accent


def _harmonize_port_color(col, is_light: bool) -> QtGui.QColor:
    """Map raw or fluorescent port colors into disciplined engineering tones matching MDAO.

    Prevents low-contrast fluorescents (pure yellow, lime green, cyan) from washing out
    against light canvas backgrounds, while maintaining semantic role recognition.
    """
    if not col or len(col) < 3:
        return QtGui.QColor(100, 116, 139, 230) if is_light else QtGui.QColor(88, 105, 128, 220)

    r, g, b = int(col[0]), int(col[1]), int(col[2])

    # Check for near neutral/gray
    if max(abs(r - g), abs(g - b), abs(r - b)) < 25:
        return QtGui.QColor(100, 116, 139, 230) if is_light else QtGui.QColor(130, 142, 160, 220)

    # Yellow / Gold (Loads, BCs, e.g. 255, 255, 0 or 255, 200, 0)
    if r > 160 and g > 150 and b < 100:
        return QtGui.QColor(175, 110, 20, 235) if is_light else QtGui.QColor(225, 165, 45, 235)

    # Bright Green / Emerald (Shapes, geometry output, e.g. 100, 255, 100)
    if g > 170 and r < 160 and b < 160:
        return QtGui.QColor(42, 128, 68, 235) if is_light else QtGui.QColor(72, 195, 115, 235)

    # Bright Cyan / Teal (Results, solver output, e.g. 0, 255, 255 or 0, 220, 255)
    if b > 180 and g > 180 and r < 100:
        return QtGui.QColor(28, 108, 168, 235) if is_light else QtGui.QColor(50, 190, 225, 235)

    # Red / Coral (Constraints, e.g. 255, 100, 100)
    if r > 180 and g < 140 and b < 140:
        return QtGui.QColor(185, 65, 55, 235) if is_light else QtGui.QColor(225, 95, 85, 235)

    # Magenta / Purple (Meshes, math, e.g. 200, 100, 200)
    if r > 160 and b > 160 and g < 140:
        return QtGui.QColor(128, 60, 145, 235) if is_light else QtGui.QColor(180, 105, 195, 235)

    # Pale / Light Blue (File / Number, e.g. 180, 220, 255)
    if b > 180 and r > 140 and g > 180:
        return QtGui.QColor(60, 105, 160, 235) if is_light else QtGui.QColor(115, 180, 235, 235)

    # Standard / other
    if is_light:
        return QtGui.QColor(int(r * 0.65), int(g * 0.65), int(b * 0.65), 235)
    return QtGui.QColor(min(255, int(r * 1.05)), min(255, int(g * 1.05)), min(255, int(b * 1.05)), 235)



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

    Paints clean, unified engineering cards aligned with MDAO's theme palette,
    with a sleek top accent line, subtle tone-on-tone header, and legible typography.
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
    radius = 12.0 if is_terminal else 8.0

    # 1. Selection Ambient Glow or Drop Shadow
    if node_item.selected:
        glow_color = QtGui.QColor(210, 153, 34, 45) if not is_light else QtGui.QColor(79, 111, 145, 50)
        for step in (4.0, 2.0):
            glow_rect = node_rect.adjusted(-step, -step, step, step)
            glow_path = QtGui.QPainterPath()
            glow_path.addRoundedRect(glow_rect, radius + step, radius + step)
            painter.fillPath(glow_path, glow_color)
    else:
        shadow_col = QtGui.QColor(15, 23, 42, 12) if is_light else QtGui.QColor(0, 0, 0, 35)
        s_rect = node_rect.adjusted(0.0, 1.5, 0.0, 2.5)
        s_path = QtGui.QPainterPath()
        s_path.addRoundedRect(s_rect, radius, radius)
        painter.fillPath(s_path, shadow_col)

    # 2. Main Node Body Fill (Unified with MDAO dark card surface)
    body_path = QtGui.QPainterPath()
    body_path.addRoundedRect(node_rect, radius, radius)

    n_col = getattr(node_item, "color", None)
    if is_light:
        if n_col and len(n_col) >= 3 and n_col[:3] not in ((13, 18, 23), (35, 35, 35)):
            body_bg = QtGui.QColor(n_col[0], n_col[1], n_col[2], 255)
        else:
            body_bg = QtGui.QColor(255, 255, 255, 255)
        if hasattr(node_item, "_set_text_color"):
            node_item._set_text_color((52, 64, 84, 255))
    else:
        body_bg = QtGui.QColor(32, 35, 41, 255)
        if hasattr(node_item, "_set_text_color"):
            node_item._set_text_color((241, 245, 249, 255))
    painter.fillPath(body_path, body_bg)

    # 3. Header Region (Subtle Tone-on-Tone Bar)
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

    badge_text, accent_color = _get_node_category_badge(node_item)
    header_bg = (
        QtGui.QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 18)
        if is_light
        else QtGui.QColor(38, 42, 49, 255)
    )

    # Header clip path (rounded top corners, square bottom)
    header_path = QtGui.QPainterPath()
    header_path.moveTo(node_rect.left() + radius, node_rect.top())
    header_path.lineTo(node_rect.right() - radius, node_rect.top())
    header_path.quadTo(node_rect.right(), node_rect.top(), node_rect.right(), node_rect.top() + radius)
    header_path.lineTo(node_rect.right(), header_rect.bottom())
    header_path.lineTo(node_rect.left(), header_rect.bottom())
    header_path.lineTo(node_rect.left(), node_rect.top() + radius)
    header_path.quadTo(node_rect.left(), node_rect.top(), node_rect.left() + radius, node_rect.top())

    painter.fillPath(header_path, header_bg)

    # Crisp top accent line (2.5px) identifying node discipline
    accent_path = QtGui.QPainterPath()
    accent_path.moveTo(node_rect.left() + radius, node_rect.top())
    accent_path.lineTo(node_rect.right() - radius, node_rect.top())
    accent_pen = QtGui.QPen(accent_color, 2.5)
    accent_pen.setCapStyle(QtCore.Qt.RoundCap)
    painter.setPen(accent_pen)
    painter.drawPath(accent_path)

    # Subtle horizontal line under header
    sep_color = QtGui.QColor(215, 220, 228, 255) if is_light else QtGui.QColor(48, 54, 64, 255)
    painter.setPen(QtGui.QPen(sep_color, 1.0))
    painter.drawLine(
        QtCore.QPointF(node_rect.left(), header_rect.bottom()),
        QtCore.QPointF(node_rect.right(), header_rect.bottom()),
    )

    # 4. Header Badge Chip (Subtle disciplined pill)
    if badge_text:
        badge_font = QtGui.QFont("Segoe UI", 7, QtGui.QFont.DemiBold)
        fm = QtGui.QFontMetrics(badge_font)
        bw = fm.horizontalAdvance(badge_text) + 10.0
        bh = 14.0
        bx = node_rect.right() - bw - 8.0
        by = node_rect.top() + (header_height - bh) / 2.0

        badge_rect = QtCore.QRectF(bx, by, bw, bh)
        badge_path = QtGui.QPainterPath()
        badge_path.addRoundedRect(badge_rect, 3.0, 3.0)

        chip_bg = (
            QtGui.QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 25)
            if is_light
            else QtGui.QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 35)
        )
        chip_border = (
            QtGui.QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 80)
            if is_light
            else QtGui.QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 90)
        )
        chip_text = (
            QtGui.QColor(52, 64, 84, 230)
            if is_light
            else QtGui.QColor(226, 232, 240, 230)
        )

        painter.fillPath(badge_path, chip_bg)
        painter.setPen(QtGui.QPen(chip_border, 1.0))
        painter.drawPath(badge_path)
        painter.setFont(badge_font)
        painter.setPen(chip_text)
        painter.drawText(badge_rect, QtCore.Qt.AlignCenter, badge_text)

    # 5. Outer Border Outline (Matching MDAO selection color and role border)
    if node_item.selected:
        border_width = 1.8
        border_color = QtGui.QColor(79, 111, 145, 255) if is_light else QtGui.QColor(210, 153, 34, 255)
    else:
        border_width = 1.2 if is_light else 1.0
        n_border = getattr(node_item, "border_color", None)
        if n_border and len(n_border) >= 3 and max(n_border[:3]) > 0 and n_border[:3] not in ((46, 57, 66), (0, 0, 0)):
            border_color = QtGui.QColor(n_border[0], n_border[1], n_border[2], 255)
        else:
            border_color = (
                QtGui.QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 160)
                if is_light
                else QtGui.QColor(48, 54, 64, 255)
            )

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

    is_light = _is_light_theme(port_item)
    is_hovered = getattr(port_item, "_hovered", False)
    is_connected = bool(getattr(port_item, "connected_pipes", None))

    port_col = getattr(port_item, "color", (200, 200, 200))
    border_col = getattr(port_item, "border_color", (240, 240, 240))

    base_color = _harmonize_port_color(port_col, is_light)
    border_color = _harmonize_port_color(border_col, is_light)

    # Ambient halo glow when hovered or connected
    if is_hovered:
        halo_col = QtGui.QColor(79, 111, 145, 90) if is_light else QtGui.QColor(210, 153, 34, 90)
        halo_rect = port_rect.adjusted(-3.0, -3.0, 3.0, 3.0)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(halo_col)
        painter.drawEllipse(halo_rect)
        border_color = QtGui.QColor(79, 111, 145, 255) if is_light else QtGui.QColor(210, 153, 34, 255)
    elif is_connected:
        halo_col = QtGui.QColor(border_color.red(), border_color.green(), border_color.blue(), 40)
        halo_rect = port_rect.adjusted(-2.0, -2.0, 2.0, 2.0)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(halo_col)
        painter.drawEllipse(halo_rect)

    # Socket outer stroke & fill
    pen = QtGui.QPen(border_color, 1.4)
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
    is_light = _is_light_theme(pipe_item)
    is_selected = getattr(pipe_item, "selected", False) or getattr(pipe_item, "_active", False)
    is_highlighted = getattr(pipe_item, "_highlight", False)
    is_disabled = getattr(pipe_item, "disabled", lambda: False)()

    # Derive connection color from the source output port if available
    out_port = getattr(pipe_item, "output_port", None)
    base_color = None
    if out_port is not None:
        p_col = getattr(out_port, "border_color", None) or getattr(out_port, "color", None)
        if p_col and len(p_col) >= 3 and max(p_col[:3]) > 30:
            base_color = _harmonize_port_color(p_col, is_light)

    if base_color is None:
        # Calibrated default slate connection matching MDAO
        base_color = QtGui.QColor(100, 116, 139, 230) if is_light else QtGui.QColor(88, 105, 128, 220)

    if is_disabled:
        pipe_color = QtGui.QColor(140, 145, 155, 120)
        pipe_width = 1.6
    elif is_selected:
        pipe_color = QtGui.QColor(79, 111, 145, 255) if is_light else QtGui.QColor(210, 153, 34, 255)
        pipe_width = 2.2
    elif is_highlighted:
        pipe_color = base_color.lighter(130) if not is_light else base_color.darker(115)
        pipe_width = 2.4
    else:
        pipe_color = base_color
        pipe_width = 1.8

    # Subtle ambient glow for selected/highlighted pipe
    if (is_selected or is_highlighted) and not is_disabled:
        glow_pen = QtGui.QPen(
            QtGui.QColor(pipe_color.red(), pipe_color.green(), pipe_color.blue(), 50),
            pipe_width + 2.5,
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

    # Sleek directional arrowhead at path midpoint (data flow direction)
    if not is_disabled and path.length() >= 18.0:
        t = 0.52
        p1 = path.pointAtPercent(max(0.0, t - 0.015))
        p2 = path.pointAtPercent(min(1.0, t + 0.015))
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len > 1e-4:
            ux = dx / seg_len
            uy = dy / seg_len
            nx = -uy
            ny = ux
            pt = path.pointAtPercent(t)
            arrow_len = 8.5
            arrow_w = 4.5
            tip = pt + QtCore.QPointF(ux * 2.5, uy * 2.5)
            left_wing = pt - QtCore.QPointF(ux * arrow_len, uy * arrow_len) + QtCore.QPointF(nx * arrow_w, ny * arrow_w)
            right_wing = pt - QtCore.QPointF(ux * arrow_len, uy * arrow_len) - QtCore.QPointF(nx * arrow_w, ny * arrow_w)
            notch = pt - QtCore.QPointF(ux * (arrow_len * 0.65), uy * (arrow_len * 0.65))

            arrow_path = QtGui.QPainterPath()
            arrow_path.moveTo(tip)
            arrow_path.lineTo(left_wing)
            arrow_path.lineTo(notch)
            arrow_path.lineTo(right_wing)
            arrow_path.closeSubpath()

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(pipe_color)
            painter.drawPath(arrow_path)

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


