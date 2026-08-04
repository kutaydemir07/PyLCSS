# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""BoundaryConditionInspectorMixin behavior for the CAD property inspector."""

from __future__ import annotations

import logging


from PySide6 import QtCore, QtWidgets


__all__ = ["BoundaryConditionInspectorMixin"]


class BoundaryConditionInspectorMixin:
    def _build_fea_bc_ui(self, node):
        """Rich Properties Panel UI for ConstraintNode, LoadNode, PressureLoadNode."""
        node_class = node.__class__.__name__

        if node_class == "ConstraintNode":
            # Use get_property (NodeGraphQt API) so we always read the live value,
            # not a potentially stale snapshot from node.model.properties.
            ct = node.get_property("constraint_type") or "Fixed"
            if ct == "Pinned (Fixed for solids)":
                ct = "Pinned"

            grp = QtWidgets.QGroupBox("Support")
            lay = QtWidgets.QFormLayout()

            combo = QtWidgets.QComboBox()
            support_labels = {
                "Fixed Support": "Fixed",
                "X Translation Fixed (UX = 0)": "Block X Translation",
                "Y Translation Fixed (UY = 0)": "Block Y Translation",
                "Z Translation Fixed (UZ = 0)": "Block Z Translation",
                "Prescribed Displacement": "Displacement",
            }
            internal_to_label = {
                value: label for label, value in support_labels.items()
            }
            combo.addItems(list(support_labels))
            current_label = internal_to_label.get(str(ct), str(ct))
            if current_label not in support_labels:
                combo.addItem(current_label)
            # Block signals while setting the initial value so construction doesn't
            # fire currentTextChanged and cause a spurious property-change loop.
            combo.blockSignals(True)
            combo.setCurrentText(current_label)
            combo.blockSignals(False)
            combo.currentTextChanged.connect(
                lambda v: self.update_property(
                    "constraint_type",
                    support_labels.get(v, v),
                )
            )
            lay.addRow("Support:", combo)

            if ct == "Displacement":
                for axis, ax in [
                    ("X", "displacement_x"),
                    ("Y", "displacement_y"),
                    ("Z", "displacement_z"),
                ]:
                    val = node.get_property(ax)
                    if val is not None:
                        enabled_prop = f"{ax}_enabled"
                        enabled_value = node.get_property(enabled_prop)
                        enabled = True if enabled_value is None else bool(enabled_value)
                        row = QtWidgets.QWidget()
                        row_layout = QtWidgets.QHBoxLayout(row)
                        row_layout.setContentsMargins(0, 0, 0, 0)
                        active = QtWidgets.QCheckBox("Prescribe")
                        active.setChecked(enabled)
                        active.toggled.connect(
                            lambda checked, p=enabled_prop: self.update_property(
                                p, checked
                            )
                        )
                        spin = QtWidgets.QDoubleSpinBox()
                        spin.setRange(-1e6, 1e6)
                        spin.setDecimals(4)
                        spin.setValue(float(val))
                        spin.valueChanged.connect(
                            lambda v, p=ax: self.update_property(p, v)
                        )
                        self.property_widgets[ax] = spin
                        row_layout.addWidget(active)
                        row_layout.addWidget(spin, 1)
                        lay.addRow(f"U{axis}:", row)

            grp.setLayout(lay)
            self.props_layout.addWidget(grp)

            # Condition expression — used when no face is connected (e.g. all .cad examples)
            cond_val = node.get_property("condition") or ""
            try:
                target_port = node.get_input("target_face")
                has_target_geometry = bool(
                    target_port and target_port.connected_ports()
                )
            except Exception:
                has_target_geometry = False
            if not has_target_geometry or cond_val:
                grp2 = QtWidgets.QGroupBox("Coordinate Filter")
                grp2.setToolTip(
                    "Boolean expression over mesh-node coordinates x, y, z. "
                    "Used only when no selection is connected."
                )
                lay2 = QtWidgets.QFormLayout()
                edit = QtWidgets.QLineEdit(str(cond_val))
                edit.setPlaceholderText("e.g. z < 0.01  or  (x < 1) & (y > 9)")
                edit.editingFinished.connect(
                    lambda: self.update_property("condition", edit.text())
                )
                lay2.addRow("Expression:", edit)
                info = QtWidgets.QLabel(
                    "Coordinates are in mm. Prefer an explicit face, edge, or "
                    "vertex selection."
                )
                info.setStyleSheet("color:#888; font-size:10px;")
                info.setWordWrap(True)
                lay2.addRow(info)
                grp2.setLayout(lay2)
                self.props_layout.addWidget(grp2)

        elif node_class == "LoadNode":
            lt = node.get_property("load_type") or "Force"
            grp = QtWidgets.QGroupBox(
                str(lt) if str(lt) in {"Force", "Gravity"} else "Force"
            )
            lay = QtWidgets.QFormLayout()

            combo = QtWidgets.QComboBox()
            combo.addItems(["Force", "Gravity"])
            combo.blockSignals(True)
            combo.setCurrentText(
                str(lt) if str(lt) in ["Force", "Gravity"] else "Force"
            )
            combo.blockSignals(False)
            combo.currentTextChanged.connect(
                lambda v: self.update_property("load_type", v)
            )
            lay.addRow("Type:", combo)

            if lt == "Force":
                fx = float(node.get_property("force_x") or 0.0)
                fy = float(node.get_property("force_y") or 0.0)
                fz = float(node.get_property("force_z") or 0.0)
                for axis, prop, val in [
                    ("X", "force_x", fx),
                    ("Y", "force_y", fy),
                    ("Z", "force_z", fz),
                ]:
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setRange(-1e12, 1e12)
                    spin.setDecimals(2)
                    spin.setValue(val)
                    spin.setSuffix(" N")
                    spin.valueChanged.connect(
                        lambda v, p=prop: self.update_property(p, v)
                    )
                    self.property_widgets[prop] = spin
                    lay.addRow(f"F{axis}:", spin)
                mag = (fx**2 + fy**2 + fz**2) ** 0.5

                # Direction arrow label  e.g. "→ (-1000, 0, 0)"
                def _dir_arrow(x, y, z):
                    dominant = max(
                        [(abs(x), "X", x), (abs(y), "Y", y), (abs(z), "Z", z)],
                        key=lambda t: t[0],
                    )
                    sign = "−" if dominant[2] < 0 else "+"
                    return f"{sign}{dominant[1]}"

                dir_lbl = QtWidgets.QLabel(
                    f"|F| {mag:.2f} N  ·  {_dir_arrow(fx, fy, fz)}"
                )
                dir_lbl.setStyleSheet(
                    "color:#6dde8d; font-weight:bold; font-size:11px;"
                )
                lay.addRow("Result:", dir_lbl)

            elif lt == "Gravity":
                grav_accel = float(node.get_property("gravity_accel") or 9810.0)
                grav_dir = node.get_property("gravity_direction") or "-Y"
                spin_g = QtWidgets.QDoubleSpinBox()
                spin_g.setRange(0, 100000)
                spin_g.setDecimals(2)
                spin_g.setValue(grav_accel)
                spin_g.setSuffix(" mm/s²")
                spin_g.valueChanged.connect(
                    lambda v: self.update_property("gravity_accel", v)
                )
                lay.addRow("Accel:", spin_g)
                cb_dir = QtWidgets.QComboBox()
                cb_dir.addItems(["-Y", "-Z", "-X", "+Y", "+Z", "+X"])
                cb_dir.blockSignals(True)
                cb_dir.setCurrentText(str(grav_dir))
                cb_dir.blockSignals(False)
                cb_dir.currentTextChanged.connect(
                    lambda v: self.update_property("gravity_direction", v)
                )
                lay.addRow("Direction:", cb_dir)

            grp.setLayout(lay)
            self.props_layout.addWidget(grp)

            # Applied-to condition expression
            cond_val = node.get_property("condition") or ""
            try:
                target_port = node.get_input("target_face")
                has_target_geometry = bool(
                    target_port and target_port.connected_ports()
                )
            except Exception:
                has_target_geometry = False
            if lt != "Gravity" and (not has_target_geometry or cond_val):
                grp2 = QtWidgets.QGroupBox("Coordinate Filter")
                grp2.setToolTip(
                    "Boolean expression over mesh-node coordinates x, y, z. "
                    "Used only when no selection is connected."
                )
                lay2 = QtWidgets.QFormLayout()
                edit = QtWidgets.QLineEdit(str(cond_val))
                edit.setPlaceholderText("e.g. z > 19  or  (abs(z) < 1.5) & (x > 9)")
                edit.editingFinished.connect(
                    lambda: self.update_property("condition", edit.text())
                )
                lay2.addRow("Expression:", edit)
                info = QtWidgets.QLabel(
                    "Coordinates are in mm. Prefer an explicit face, edge, or "
                    "vertex selection."
                )
                info.setStyleSheet("color:#888; font-size:10px;")
                info.setWordWrap(True)
                lay2.addRow(info)
                grp2.setLayout(lay2)
                self.props_layout.addWidget(grp2)

        # Preview is shared by support, force/gravity, and pressure.
        if node_class in ("ConstraintNode", "LoadNode", "PressureLoadNode"):
            btn_preview = QtWidgets.QPushButton("Preview in 3D")
            btn_preview.setToolTip(
                "Run the CAD graph without launching a solver and highlight "
                "the selected face, edge, or vertex. A coordinate-expression "
                "condition has no exact CAD entity to highlight."
            )
            btn_preview.setStyleSheet(
                "QPushButton {"
                "  background: #1e5aab; color: white; border-radius: 4px;"
                "  padding: 5px 10px; font-weight: bold; font-size: 12px;"
                "  margin-top: 6px;"
                "}"
                "QPushButton:hover { background: #2673cc; }"
            )

            def _on_preview(checked=False, _node=node):
                app = self._get_main_app()
                if app is None:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "No viewer",
                        "Cannot reach the main CAD widget; preview unavailable.",
                    )
                    return

                # 1. Render the upstream geometry first.  Many bugs reported as
                # "Preview does nothing" turn out to be that the user never ran
                # the graph — _last_result is None on every upstream node, so
                # the viewer has no shape to draw the overlay on.
                source, renderable = app._get_render_context_for_node(_node)
                if renderable is None:
                    app._last_rendered_node = _node
                    try:
                        app._execute_graph(skip_simulation=True)
                    except Exception as exc:
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Preview Failed",
                            str(exc),
                        )
                        return
                    source, renderable = app._get_render_context_for_node(_node)
                if renderable is not None:
                    app._render_result_in_viewer(renderable)
                else:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Nothing to preview",
                        "Connect geometry first.",
                    )
                    return

                # 2. Now draw the BC overlay on top of the shape.
                try:
                    app._show_bc_for_node(_node)
                except Exception as exc:
                    # _show_bc_for_node has its own try/except; this is belt-
                    # and-suspenders.  We still want the user to see *why*.
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Preview Failed",
                        f"Could not draw the selection: {exc}",
                    )
                    return

                # 3. Verify the overlay actually had something to draw.
                # _collect_bc_for_node is cheap; surface a friendly hint when
                # the overlay was a no-op (e.g. the BC uses a condition
                # string that doesn't match any nodes).
                try:
                    c_faces, l_faces, l_vecs = app._collect_bc_for_node(_node)
                    if not (c_faces or l_faces or l_vecs):
                        sb = getattr(app, "statusBar", lambda: None)()
                        if sb is not None:
                            sb.showMessage(
                                "No selected geometry to highlight.",
                                6000,
                            )
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )

            btn_preview.clicked.connect(_on_preview)
            self.props_layout.addWidget(btn_preview)

        if node_class == "PressureLoadNode":
            grp = QtWidgets.QGroupBox("Pressure")
            lay = QtWidgets.QFormLayout()

            raw_pressure = node.get_property("pressure")
            pval = float(1.0 if raw_pressure is None else raw_pressure)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.000001, 1e9)
            spin.setDecimals(4)
            spin.blockSignals(True)
            spin.setValue(pval)
            spin.blockSignals(False)
            spin.setSuffix(" MPa")
            spin.valueChanged.connect(lambda v: self.update_property("pressure", v))
            self.property_widgets["pressure"] = spin
            lay.addRow("Pressure:", spin)

            pdir = node.get_property("direction") or "Inward"
            dir_combo = QtWidgets.QComboBox()
            dir_combo.addItems(["Inward", "Outward"])
            dir_combo.blockSignals(True)
            dir_combo.setCurrentText(
                str(pdir) if str(pdir) in ["Outward", "Inward"] else "Inward"
            )
            dir_combo.blockSignals(False)
            dir_combo.currentTextChanged.connect(
                lambda v: self.update_property("direction", v)
            )
            lay.addRow("Direction:", dir_combo)

            grp.setLayout(lay)
            self.props_layout.insertWidget(
                max(0, self.props_layout.count() - 1),
                grp,
            )

        elif node_class not in ("ConstraintNode", "LoadNode"):
            self._build_generic_ui(node)

    def update_property(self, prop_name, value):
        """Update node property and mark as dirty for recalculation."""
        if self.current_node:
            try:
                self._updating_property = True
                old = self.current_node.get_property(prop_name)
                self.current_node.set_property(prop_name, value)
                # Mark node as dirty for recalculation
                if hasattr(self.current_node, "_last_hash"):
                    self.current_node._last_hash = None  # Invalidate hash cache
                try:
                    self.property_changed.emit(self.current_node, prop_name, old, value)
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
                try:
                    from pylcss.design_studio.core.port_schema import (
                        apply_context_port_visibility,
                    )

                    apply_context_port_visibility(self.current_node)
                    self.current_node.view.draw_node()
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Could not refresh contextual ports.", exc_info=True
                    )
                if prop_name in {
                    "application_scope",
                    "enable_fracture",
                    "enable_mass_scaling",
                    "load_type",
                    "mesh_type",
                }:
                    selected_node = self.current_node
                    QtCore.QTimer.singleShot(
                        0,
                        lambda n=selected_node: (
                            self.display_node(n)
                            if self.current_node is n
                            else None
                        ),
                    )
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            finally:
                self._updating_property = False
