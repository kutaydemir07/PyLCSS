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
                lambda v: (
                    self.update_property("constraint_type", support_labels.get(v, v)),
                    QtCore.QTimer.singleShot(
                        0,
                        lambda n=node: self.display_node(n) if self.current_node is n else None,
                    ),
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

        elif node_class == "LoadNode":
            lt = node.get_property("load_type") or "Force"
            grp = QtWidgets.QGroupBox(
                str(lt) if str(lt) in {"Force", "Pressure", "Gravity"} else "Load"
            )
            lay = QtWidgets.QFormLayout()

            combo = QtWidgets.QComboBox()
            combo.addItems(["Force", "Pressure", "Gravity"])
            combo.blockSignals(True)
            combo.setCurrentText(
                str(lt) if str(lt) in ["Force", "Pressure", "Gravity"] else "Force"
            )
            combo.blockSignals(False)
            combo.currentTextChanged.connect(
                lambda v: (
                    self.update_property("load_type", v),
                    QtCore.QTimer.singleShot(
                        0,
                        lambda n=node: self.display_node(n) if self.current_node is n else None,
                    ),
                )
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

            elif lt == "Pressure":
                press = float(node.get_property("pressure") or 1.0)
                direct = str(node.get_property("direction") or "Inward")
                spin_p = QtWidgets.QDoubleSpinBox()
                spin_p.setRange(0.001, 1e8)
                spin_p.setDecimals(3)
                spin_p.setValue(press)
                spin_p.setSuffix(" MPa")
                spin_p.valueChanged.connect(
                    lambda v: self.update_property("pressure", v)
                )
                self.property_widgets["pressure"] = spin_p
                lay.addRow("Pressure:", spin_p)

                combo_dir = QtWidgets.QComboBox()
                combo_dir.addItems(["Inward (Compression)", "Outward (Tension)"])
                combo_dir.blockSignals(True)
                combo_dir.setCurrentText(
                    "Inward (Compression)" if direct == "Inward" else "Outward (Tension)"
                )
                combo_dir.blockSignals(False)
                combo_dir.currentTextChanged.connect(
                    lambda v: self.update_property(
                        "direction",
                        "Inward" if "Inward" in v else "Outward",
                    )
                )
                self.property_widgets["direction"] = combo_dir
                lay.addRow("Direction:", combo_dir)

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

        elif node_class == "PressureLoadNode":
            grp = QtWidgets.QGroupBox("Pressure")
            lay = QtWidgets.QFormLayout()
            press = float(node.get_property("pressure") or 1.0)
            direct = str(node.get_property("direction") or "Inward")
            spin_p = QtWidgets.QDoubleSpinBox()
            spin_p.setRange(0.001, 1e8)
            spin_p.setDecimals(3)
            spin_p.setValue(press)
            spin_p.setSuffix(" MPa")
            spin_p.valueChanged.connect(
                lambda v: self.update_property("pressure", v)
            )
            self.property_widgets["pressure"] = spin_p
            lay.addRow("Pressure:", spin_p)

            combo_dir = QtWidgets.QComboBox()
            combo_dir.addItems(["Inward (Compression)", "Outward (Tension)"])
            combo_dir.blockSignals(True)
            combo_dir.setCurrentText(
                "Inward (Compression)" if direct == "Inward" else "Outward (Tension)"
            )
            combo_dir.blockSignals(False)
            combo_dir.currentTextChanged.connect(
                lambda v: self.update_property(
                    "direction",
                    "Inward" if "Inward" in v else "Outward",
                )
            )
            self.property_widgets["direction"] = combo_dir
            lay.addRow("Direction:", combo_dir)
            grp.setLayout(lay)
            self.props_layout.addWidget(grp)

        # Preview is shared by support, force/gravity, and pressure.
        if node_class in ("ConstraintNode", "LoadNode", "PressureLoadNode"):
            btn_preview = QtWidgets.QPushButton("Preview in 3D")
            btn_preview.setToolTip(
                "Run the CAD graph without launching a solver and highlight "
                "the selected face, edge, or vertex."
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
                if (
                    self.current_node.has_property("advanced_overrides")
                    and prop_name in getattr(self, "_topopt_advanced_property_names", set())
                    and not getattr(self, "_applying_topopt_defaults", False)
                ):
                    import json

                    overrides = set(json.loads(self.current_node.get_property("advanced_overrides") or "[]"))
                    overrides.add(prop_name)
                    self.current_node.set_property("advanced_overrides", json.dumps(sorted(overrides)))
                if self.current_node.__class__.__name__ in {'LatticeInfillNode'} and prop_name in {
                    'structure_mode', 'lattice_target_relative_density', 'infill_cell_size_mode',
                    'infill_cell_size_mm', 'infill_fineness', 'infill_skin_thickness_mm', 'infill_build_quality',
                    'volfrac', 'lattice_cell_size_mm', 'lattice_settings_mode', 'lattice_member_thickness_mm',
                    'lattice_variable_density', 'lattice_min_relative_density', 'lattice_max_relative_density'}:
                    node = self.current_node
                    timer = getattr(self, '_infill_build_timer', None)
                    if timer is not None:
                        timer.stop()
                        timer.deleteLater()
                    timer = QtCore.QTimer(self)
                    timer.setSingleShot(True)
                    timer.timeout.connect(lambda: self._build_and_show_lattice_infill(node))
                    self._infill_build_timer = timer
                    timer.start(350)
                # Mark node as dirty for recalculation
                from pylcss.cae.design_studio.topology_optimization.configuration.fea_meshing import (
                    is_topopt_fea_setting,
                )
                if hasattr(self.current_node, "_last_hash") and prop_name not in {
                    "advanced_settings_visible", "advanced_overrides",
                } and not is_topopt_fea_setting(self.current_node, prop_name):
                    self.current_node._last_hash = None  # Invalidate hash cache
                try:
                    self.property_changed.emit(self.current_node, prop_name, old, value)
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
                try:
                    from pylcss.cae.design_studio.core.port_schema import (
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
