# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""GenericInspectorMixin behavior for the CAD property inspector."""

from __future__ import annotations

import logging

import os

from PySide6 import QtCore, QtWidgets


from .inspector_controls import ExpressionEdit

__all__ = ["GenericInspectorMixin"]


class GenericInspectorMixin:
    @classmethod
    def _human_property_label(cls, property_name):
        """Return a readable inspector label with no implementation syntax."""
        explicit = cls._PROPERTY_LABELS.get(property_name)
        if explicit:
            return explicit
        words = []
        for token in str(property_name or "").strip("_").split("_"):
            if not token:
                continue
            words.append(cls._PROPERTY_ACRONYMS.get(token.lower(), token.title()))
        return " ".join(words) or "Property"

    @classmethod
    def _section_for(cls, prop_name):
        for title, prefixes in cls._PROPERTY_SECTIONS:
            for pref in prefixes:
                if prop_name == pref or prop_name.startswith(pref):
                    return title
        return "General"

    @classmethod
    def _is_directory_prop(cls, name):
        n = name.lower()
        return n.endswith("_dir") or n == "work_dir" or n.endswith("_directory")

    @classmethod
    def _looks_like_path_prop(cls, name):
        n = name.lower()
        if cls._is_directory_prop(name):
            return True
        # Match anything that ends with _path, equals 'filepath', or is one
        # of our explicitly known file properties.
        if n.endswith("_path") or n == "filepath":
            return True
        for sub, _, _ in cls._PATH_PROP_FILTERS:
            if sub in n:
                return True
        return False

    def _build_path_widget(self, name, val):
        """QLineEdit + Browse button for path / directory properties."""
        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        edit = QtWidgets.QLineEdit(str(val) if val is not None else "")
        edit.setPlaceholderText(
            "Browse to a folder…"
            if self._is_directory_prop(name)
            else "Browse to a file…"
        )
        edit.editingFinished.connect(
            lambda n=name, w=edit: self.update_property(n, w.text())
        )
        h.addWidget(edit, 1)

        btn = QtWidgets.QToolButton()
        btn.setText("...")
        btn.setToolTip(
            "Pick a folder" if self._is_directory_prop(name) else "Pick a file"
        )
        btn.setMinimumWidth(28)

        def browse():
            current = edit.text().strip()
            start_dir = (
                current
                if (current and os.path.isdir(os.path.dirname(current) or current))
                else ""
            )
            if self._is_directory_prop(name):
                chosen = QtWidgets.QFileDialog.getExistingDirectory(
                    self, "Select directory", start_dir or current
                )
            else:
                title = "Select file"
                filt = "All files (*)"
                for sub, t, f in self._PATH_PROP_FILTERS:
                    if sub in name.lower():
                        title, filt = t, f
                        break
                chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, title, start_dir or current, filt
                )
            if chosen:
                edit.setText(chosen)
                self.update_property(name, chosen)

        btn.clicked.connect(browse)
        h.addWidget(btn)
        return container

    def _build_generic_ui(self, node):
        """Generic Clean UI - uses node.properties() for custom properties.

        Properties are grouped into semantic sections (External Solver,
        Visualization, Solver, Material, Mesh, Geometry, Load, Constraint,
        General) and rendered with the right widget per type — sliders when
        the node declared a range, checkboxes for bools, combos when items
        are declared, expression-aware edits for numerics.
        """
        try:
            all_props = node.properties()
            props = all_props.get("custom", {})
        except Exception:
            props = {}

        if not props:
            lbl = QtWidgets.QLabel("No editable properties")
            lbl.setStyleSheet("color: #888; font-style: italic;")
            self.props_layout.addWidget(lbl)
            return

        # Collect NodeGraphQt's per-property metadata once.  Items and ranges
        # both live on the model — items are how we know it's a combo, range
        # is how we pick a slider over a spinbox.
        prop_attrs = {}
        try:
            if hasattr(node, "model"):
                model = node.model
                temp_attrs = getattr(model, "_TEMP_property_attrs", None) or {}
                if isinstance(temp_attrs, dict):
                    prop_attrs.update(
                        {
                            k: dict(v)
                            for k, v in temp_attrs.items()
                            if isinstance(v, dict)
                        }
                    )
                if getattr(model, "_graph_model", None) is not None:
                    try:
                        common = (
                            model._graph_model.get_node_common_properties(model.type_)
                            or {}
                        )
                        for k, v in common.items():
                            if isinstance(v, dict):
                                merged = prop_attrs.get(k, {})
                                merged.update(v)
                                prop_attrs[k] = merged
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

        # Skip empty/optional properties so the inspector isn't a wall of blanks.
        def _should_skip(name, val):
            if name.startswith("_"):
                return True
            if name in self._PROPERTY_HIDE_ALWAYS:
                return True
            if name in self._PROPERTY_HIDE_IF_EMPTY:
                if val is None or val == "" or val == 0 or val == 0.0:
                    return True
            # Show only settings that the selected mesh backend/formulation
            # actually consumes. This prevents inactive numerical controls from
            # looking like they affect the generated mesh.
            if node.__class__.__name__ == "MeshNode":
                mesh_type = str(props.get("mesh_type") or "Tet")
                mesher = str(props.get("mesher") or "Gmsh (Recommended)")
                if name in {
                    "gmsh_algorithm",
                    "curvature_elements",
                } and not mesher.startswith("Gmsh"):
                    return True
                if name in {"shell_thickness", "shell_nip"} and mesh_type != "Shell":
                    return True
            if node.__class__.__name__ == "ConstraintNode":
                is_displacement = (
                    str(props.get("constraint_type") or "Fixed") == "Displacement"
                )
                if name.startswith("displacement_") and not is_displacement:
                    return True
            if node.__class__.__name__ == "LoadNode":
                is_gravity = str(props.get("load_type") or "Force") == "Gravity"
                if name.startswith("force_") and is_gravity:
                    return True
                if name.startswith("gravity_") and not is_gravity:
                    return True
            if node.__class__.__name__ == "CrashMaterialNode":
                if name == "failure_strain" and not bool(props.get("enable_fracture")):
                    return True
            return False

        # Bucket properties into sections, preserving the section order above.
        section_order = [t for t, _ in self._PROPERTY_SECTIONS] + ["General"]
        buckets = {title: [] for title in section_order}
        for name in sorted(props.keys()):
            val = props[name]
            if _should_skip(name, val):
                continue
            buckets[self._section_for(name)].append((name, val))

        # Hard-coded fallback combo items for nodes whose ``items`` metadata
        # is missing from NodeGraphQt's per-instance attrs (happens for older
        # saved files and for properties created via add_text_input()).  The
        # entry whose superset value list covers the saved string wins.
        known_combos = {
            "operation": ["Union", "Cut", "Intersect"],
            "preset": [
                "Custom",
                "Steel (Structural)",
                "Steel (Stainless 304)",
                "Aluminum 6061-T6",
                "Aluminum 7075-T6",
                "Titanium Ti-6Al-4V",
                "Copper (Annealed)",
                "Brass",
                "Cast Iron (Gray)",
                "Magnesium AZ31",
                "Nickel Alloy 718",
                "CFRP (Quasi-Isotropic)",
                "GFRP (E-Glass)",
                "Concrete (Normal)",
                "ABS Plastic",
                "Nylon 6/6",
                "PEEK",
                "Wood (Oak)",
            ],
            "mesh_type": ["Tet", "Tet10", "Shell"],
            "constraint_type": [
                "Fixed",
                "Block X Translation",
                "Block Y Translation",
                "Block Z Translation",
                "Displacement",
            ],
            "load_type": ["Force", "Gravity", "Pressure"],
            "gravity_direction": ["-Y", "-Z", "-X", "+Y", "+Z", "+X"],
            # Union of every solver node's visualization vocabulary.
            "visualization": [
                "Von Mises Stress",
                "Displacement",
                "Plastic Strain",
                "Failed Elements",
                "Density",
                "Recovered Shape",
            ],
            "cad_reconstruction_method": [
                "Recovered Shape",
            ],
            "source_geometry": [
                "Recovered Shape",
            ],
            "filter_type": ["sensitivity", "density"],
            "update_scheme": ["MMA", "OC"],
            # New combos introduced by the CalculiX / OpenRadioss rewrites.
            "analysis_type": ["Linear", "Nonlinear (Geometric)", "Nonlinear (Plastic)"],
            "deformation_scale": ["Auto", "1x", "5x", "10x", "50x", "100x", "200x"],
            "application_scope": [
                "Fixed specimen + moving impactor",
                "Moving body + fixed wall",
                "Prescribed moving wall",
            ],
            # Optimisation-node combos that were always there but worth pinning.
            "objective": [
                "Min Weight",
                "Min Compliance",
                "Min Max Stress",
                "Uniform Stress",
            ],
            "optimizer": [
                "COBYLA",
                "Nelder-Mead",
                "SLSQP",
                "L-BFGS-B",
                "trust-constr",
                "Powell",
            ],
            "sensitivity_method": ["Biological Stress Leveling", "Adjoint Compliance"],
            "element_type": ["Fast (Linear P1)", "Accurate (Quadratic P2)"],
        }
        if node.__class__.__name__ == "CrashMaterialNode":
            try:
                from pylcss.design_studio.crash.materials import CRASH_MATERIAL_PRESETS

                known_combos["preset"] = list(CRASH_MATERIAL_PRESETS.keys())
            except Exception:
                known_combos["preset"] = [
                    "Custom",
                    "Steel (Structural A36)",
                    "Steel (High-Strength DP780)",
                    "DP780 Dual-Phase",
                    "Steel (Ultra-High UHSS 1500)",
                    "Aluminum 6061-T6",
                    "Aluminum 5052-H32 (Crush)",
                    "CFRP (Quasi-Isotropic, proxy)",
                ]
        rendered_any = False
        for title in section_order:
            entries = buckets[title]
            if not entries:
                continue
            rendered_any = True
            group = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QFormLayout()
            layout.setLabelAlignment(QtCore.Qt.AlignRight)
            layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

            for name, val in entries:
                label_text = self._human_property_label(name)
                attrs = prop_attrs.get(name, {})
                combo_items = attrs.get("items") if isinstance(attrs, dict) else None
                if name == "application_scope":
                    combo_items = known_combos[name]
                elif not combo_items and name in known_combos:
                    combo_items = known_combos[name]
                value_range = attrs.get("range") if isinstance(attrs, dict) else None
                tooltip = attrs.get("tooltip") if isinstance(attrs, dict) else None

                if combo_items:
                    widget = QtWidgets.QComboBox()
                    widget.setMinimumContentsLength(8)
                    widget.setSizeAdjustPolicy(
                        QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
                    )
                    item_texts = [str(item) for item in combo_items]
                    if val is not None and str(val) not in item_texts:
                        item_texts.append(str(val))
                    widget.addItems(item_texts)
                    if val is not None:
                        idx = widget.findText(str(val))
                        if idx >= 0:
                            widget.setCurrentIndex(idx)
                    widget.currentTextChanged.connect(
                        lambda v, n=name: self.update_property(n, v)
                    )
                elif isinstance(val, bool):
                    widget = QtWidgets.QCheckBox()
                    widget.setChecked(val)
                    widget.stateChanged.connect(
                        lambda s, n=name: self.update_property(n, bool(s))
                    )
                elif (
                    isinstance(val, (int, float))
                    and value_range
                    and len(value_range) == 2
                ):
                    # Range-bounded numerics get a slider+spin combo for clarity.
                    lo, hi = float(value_range[0]), float(value_range[1])
                    is_int = (
                        isinstance(val, int)
                        and float(lo).is_integer()
                        and float(hi).is_integer()
                    )
                    container = QtWidgets.QWidget()
                    h = QtWidgets.QHBoxLayout(container)
                    h.setContentsMargins(0, 0, 0, 0)
                    h.setSpacing(6)
                    if is_int:
                        spin = QtWidgets.QSpinBox()
                        spin.setRange(int(lo), int(hi))
                        spin.setValue(int(val))
                    else:
                        spin = QtWidgets.QDoubleSpinBox()
                        spin.setDecimals(4)
                        spin.setRange(lo, hi)
                        spin.setValue(float(val))
                    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                    slider_steps = 1000 if not is_int else max(1, int(hi - lo))
                    slider.setRange(0, slider_steps)

                    def _to_slider(v, lo=lo, hi=hi, steps=slider_steps):
                        if hi <= lo:
                            return 0
                        return int(round((v - lo) / (hi - lo) * steps))

                    def _from_slider(s, lo=lo, hi=hi, steps=slider_steps):
                        return lo + (s / steps) * (hi - lo)

                    slider.setValue(_to_slider(float(val)))

                    def on_slider(s, n=name, sp=spin, conv=_from_slider, is_int=is_int):
                        v = conv(s)
                        if is_int:
                            v = int(round(v))
                        sp.blockSignals(True)
                        sp.setValue(v)
                        sp.blockSignals(False)
                        self.update_property(n, v)

                    def on_spin(v, n=name, sl=slider, conv=_to_slider):
                        sl.blockSignals(True)
                        sl.setValue(conv(float(v)))
                        sl.blockSignals(False)
                        self.update_property(n, v)

                    slider.valueChanged.connect(on_slider)
                    spin.valueChanged.connect(on_spin)
                    h.addWidget(slider, 1)
                    h.addWidget(spin)
                    widget = container
                elif isinstance(val, (int, float)):
                    widget = ExpressionEdit(val)
                    widget.value_changed.connect(
                        lambda v, n=name: self.update_property(n, v)
                    )
                elif self._looks_like_path_prop(name):
                    widget = self._build_path_widget(name, val)
                elif val is not None:
                    widget = QtWidgets.QLineEdit(str(val))
                    widget.editingFinished.connect(
                        lambda n=name, w=widget: self.update_property(n, w.text())
                    )
                else:
                    continue

                widget.setMinimumWidth(0)
                widget.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding,
                    QtWidgets.QSizePolicy.Fixed,
                )
                # Fall back to the panel's static tooltip table when the
                # node itself didn't ship one.  Lets us document the meaning
                # of obscure knobs without touching every node class.
                if not tooltip:
                    tooltip = self._PROPERTY_TOOLTIPS.get(name)
                if tooltip:
                    widget.setToolTip(str(tooltip))
                layout.addRow(label_text, widget)

            group.setLayout(layout)
            self.props_layout.addWidget(group)

        if not rendered_any:
            lbl = QtWidgets.QLabel("No editable properties")
            lbl.setStyleSheet("color: #888; font-style: italic;")
            self.props_layout.addWidget(lbl)
