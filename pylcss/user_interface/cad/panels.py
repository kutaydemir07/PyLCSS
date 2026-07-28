# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Timeline, results, and node-library panels for Design Studio."""

from __future__ import annotations

import logging

from datetime import datetime

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QDrag

__all__ = ["LibraryPanel", "ResultsPanel", "TimelinePanel"]


class TimelinePanel(QtWidgets.QWidget):
    """Timeline/History panel for tracking changes."""

    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Timeline")
        title.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        self.layout.addWidget(title)

        self.history_list = QtWidgets.QListWidget()
        self.layout.addWidget(self.history_list)

    def add_event(self, event_text):
        """Add an event to the timeline."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_list.addItem(f"[{timestamp}] {event_text}")
        # Scroll to bottom
        self.history_list.scrollToBottom()


class ResultsPanel(QtWidgets.QWidget):
    """Summary of the most recent FEA / Crash / TopOpt solve.

    Pulled from the result dict that the solver nodes already produce — so we
    do not duplicate any computation; this is purely a presentation surface
    for what otherwise only goes to stdout.
    """

    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            """
            QGroupBox { font-weight: bold; margin-top: 8px; padding-top: 10px; border: 1px solid #444; border-radius: 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #BBDEFB; }
            QLabel.metric-key { color: #B0BEC5; }
            QLabel.metric-val { color: #FAFAFA; font-weight: bold; }
            """
        )
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        self._empty = QtWidgets.QLabel(
            "No solver results yet — run an FEA, Crash, or Topology node."
        )
        outer.addWidget(self._empty)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        self._scroll.setWidget(self._content)
        self._scroll.setVisible(False)
        outer.addWidget(self._scroll, 1)

    @staticmethod
    def _fmt(value, unit=""):
        if value is None:
            return "—"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)
        if v == 0.0:
            return f"0 {unit}".strip()
        if abs(v) >= 1e3 or abs(v) < 1e-2:
            return f"{v:.3e} {unit}".strip()
        return f"{v:.4g} {unit}".strip()

    def _clear(self):
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.deleteLater()

    def _add_section(self, title, rows):
        """rows: list of (label, value-string)."""
        group = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(group)
        form.setContentsMargins(10, 6, 10, 10)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(4)
        for label, val in rows:
            lk = QtWidgets.QLabel(label)
            lk.setProperty("class", "metric-key")
            lk.setStyleSheet("color: #B0BEC5;")
            lv = QtWidgets.QLabel(val)
            lv.setStyleSheet("color: #FAFAFA; font-weight: bold;")
            lv.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            form.addRow(lk, lv)
        self._content_layout.addWidget(group)

    def _add_warnings(self, warnings):
        if not warnings:
            return
        group = QtWidgets.QGroupBox(f"Warnings ({len(warnings)})")
        v = QtWidgets.QVBoxLayout(group)
        v.setContentsMargins(10, 6, 10, 10)
        for w in warnings:
            label = QtWidgets.QLabel(f"• {w}")
            label.setWordWrap(True)
            label.setStyleSheet("color: #FFCC80;")
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            v.addWidget(label)
        self._content_layout.addWidget(group)

    def show_result(self, data):
        """Populate from a solver result dict.  Safe to call with None."""
        if not isinstance(data, dict) or "type" not in data:
            self._empty.setVisible(True)
            self._scroll.setVisible(False)
            return
        rtype = data.get("type")
        if rtype not in (
            "fea",
            "crash",
            "external_solver",
            "topopt_voxel",
            "mesh",
            "remesh",
        ):
            self._empty.setVisible(True)
            self._scroll.setVisible(False)
            return

        self._clear()
        self._empty.setVisible(False)
        self._scroll.setVisible(True)

        backend = data.get("backend") or (
            "CalculiX" if rtype == "fea" else "OpenRadioss" if rtype == "crash" else "—"
        )
        if rtype == "topopt_voxel":
            backend = data.get("backend") or "pyMOTO"
        meta_rows = [
            ("Type", rtype.upper()),
            ("Backend", str(backend)),
        ]
        if "visualization_mode" in data:
            meta_rows.append(("Visualization", str(data["visualization_mode"])))
        if "analysis_type" in data:
            meta_rows.append(("Analysis", str(data["analysis_type"])))
        if "external_status" in data:
            meta_rows.append(("Solver status", str(data["external_status"])))
        if "work_dir" in data:
            meta_rows.append(("Work directory", str(data["work_dir"])))
        self._add_section("Run", meta_rows)

        mesh_obj = data.get("mesh")
        quality = data.get("mesh_quality")
        if quality is None and mesh_obj is not None:
            quality = getattr(mesh_obj, "quality_report", None)
        if isinstance(quality, dict):
            mesh_rows = []
            if mesh_obj is not None:
                try:
                    mesh_rows.extend(
                        [
                            ("Nodes", f"{int(np.asarray(mesh_obj.p).shape[1]):,}"),
                            ("Elements", f"{int(np.asarray(mesh_obj.t).shape[1]):,}"),
                        ]
                    )
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            mesh_rows.extend(
                [
                    ("Assessment", str(quality.get("assessment") or "—")),
                    (
                        "Mean-ratio quality (minimum)",
                        self._fmt(quality.get("min_mean_ratio")),
                    ),
                    (
                        "Mean-ratio quality (5th percentile)",
                        self._fmt(quality.get("p05_mean_ratio")),
                    ),
                    (
                        "Mean-ratio quality (mean)",
                        self._fmt(quality.get("mean_mean_ratio")),
                    ),
                    (
                        "Maximum edge ratio",
                        self._fmt(quality.get("max_edge_ratio")),
                    ),
                    (
                        "Collapsed/invalid elements",
                        str(int(quality.get("degenerate_elements") or 0)),
                    ),
                ]
            )
            self._add_section("Mesh quality (1.0 = ideal)", mesh_rows)

        if rtype == "fea" or rtype == "external_solver":
            metrics = []
            peak_disp = data.get("peak_displacement")
            if peak_disp is None and data.get("displacement") is not None:
                try:
                    import numpy as np

                    arr = np.asarray(data["displacement"], dtype=float)
                    if arr.size:
                        if arr.ndim == 1 and arr.size % 3 == 0:
                            arr = arr.reshape((-1, 3))
                        peak_disp = float(np.max(np.linalg.norm(arr, axis=1)))
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            if peak_disp is not None:
                metrics.append(("Peak displacement", self._fmt(peak_disp, "mm")))

            peak_stress = data.get("peak_stress_nodal")
            if peak_stress is None and data.get("stress") is not None:
                try:
                    import numpy as np

                    arr = np.asarray(data["stress"], dtype=float)
                    if arr.size:
                        peak_stress = float(np.max(arr))
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            if peak_stress is not None:
                metrics.append(
                    ("Peak stress (nodal extrapolated)", self._fmt(peak_stress, "MPa"))
                )
            if data.get("stress_location") == "gauss" and "max_stress_gauss" in data:
                metrics.append(
                    ("Peak stress (Gauss)", self._fmt(data["max_stress_gauss"], "MPa"))
                )
            if "strain_energy" in data:
                metrics.append(
                    ("Strain energy", self._fmt(data["strain_energy"], "N mm"))
                )
            if data.get("compliance") is not None:
                metrics.append(("Compliance", self._fmt(data["compliance"], "N mm")))
            if data.get("reaction_force") is not None:
                try:
                    reaction = np.asarray(data["reaction_force"], dtype=float).reshape(
                        -1
                    )
                    if reaction.size >= 3:
                        metrics.append(
                            (
                                "Support reaction (X, Y, Z)",
                                ", ".join(
                                    self._fmt(value, "N") for value in reaction[:3]
                                ),
                            )
                        )
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            if "volume" in data:
                metrics.append(("Volume", self._fmt(data["volume"], "mm^3")))
            if "mass" in data:
                metrics.append(("Mass", self._fmt(float(data["mass"]) * 1000.0, "kg")))
            if "deformation_scale" in data:
                try:
                    raw_scale = data["deformation_scale"]
                    if isinstance(raw_scale, str):
                        text_scale = raw_scale.strip().lower()
                        scale = (
                            1.0
                            if text_scale == "auto"
                            else float(text_scale.rstrip("x"))
                        )
                    else:
                        scale = float(raw_scale)
                    metrics.append(("Deformation scale", f"{scale:.1f}×"))
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            if metrics:
                self._add_section("Result", metrics)

        if rtype == "crash":
            crash_rows = []
            if "peak_displacement" in data:
                crash_rows.append(
                    (
                        "Event peak displacement",
                        self._fmt(data["peak_displacement"], "mm"),
                    )
                )
            if "final_frame_displacement" in data:
                crash_rows.append(
                    (
                        "Final-frame displacement",
                        self._fmt(data["final_frame_displacement"], "mm"),
                    )
                )
            if "peak_stress" in data:
                crash_rows.append(
                    ("Event peak Von Mises", self._fmt(data["peak_stress"], "MPa"))
                )
            if "final_frame_stress" in data:
                crash_rows.append(
                    (
                        "Final-frame Von Mises",
                        self._fmt(data["final_frame_stress"], "MPa"),
                    )
                )
            if "absorbed_energy_kj" in data:
                crash_rows.append(
                    (
                        "Absorbed / internal energy",
                        self._fmt(data["absorbed_energy_kj"], "kJ"),
                    )
                )
            elif "absorbed_energy" in data:
                crash_rows.append(
                    (
                        "Absorbed / internal energy",
                        self._fmt(float(data["absorbed_energy"]) / 1.0e6, "kJ"),
                    )
                )
            if "n_failed" in data:
                crash_rows.append(("Failed elements", str(data["n_failed"])))
            for key, label, unit in (
                ("peak_force", "Peak crushing force", "kN"),
                ("mean_force", "Mean crushing force", "kN"),
                ("crush_force_efficiency", "Crush-force efficiency", ""),
                ("specific_energy_absorption", "Specific energy absorption", "kJ/kg"),
                ("crush_distance", "Useful crush distance", "mm"),
                ("peak_acceleration_g", "Peak crash-pulse acceleration", "g"),
                ("delta_v", "Crash-pulse velocity change", "m/s"),
            ):
                if data.get(key) is not None:
                    crash_rows.append((label, self._fmt(data[key], unit)))
            if data.get("quality_status"):
                crash_rows.append(("Qualification", str(data["quality_status"])))
            if "ml_eligible" in data:
                crash_rows.append(
                    ("Eligible for ML dataset", "Yes" if data["ml_eligible"] else "No")
                )
            if "frames" in data and data["frames"]:
                crash_rows.append(("Animation frames", str(len(data["frames"]))))
            if "energy_balance_max_error" in data:
                crash_rows.append(
                    (
                        "Max. |energy error|",
                        f"{float(data['energy_balance_max_error']) * 100:.1f}%",
                    )
                )
            if "mass_balance_max_error" in data:
                crash_rows.append(
                    (
                        "Max. |mass change|",
                        f"{float(data['mass_balance_max_error']) * 100:.2f}%",
                    )
                )
            if crash_rows:
                self._add_section("Crash result", crash_rows)

        if rtype == "topopt_voxel":
            topo_rows = []
            if data.get("design_goal"):
                topo_rows.append(("Goal", str(data["design_goal"])))
            if data.get("target_vol_frac") is not None:
                topo_rows.append(
                    ("Material budget", f"{float(data['target_vol_frac']) * 100:.1f}%")
                )
            if data.get("final_vol_frac") is not None:
                topo_rows.append(
                    ("Final material", f"{float(data['final_vol_frac']) * 100:.1f}%")
                )
            if data.get("compliance") is not None:
                topo_rows.append(("Compliance", self._fmt(data["compliance"], "N mm")))
            if data.get("stress_pnorm") is not None:
                topo_rows.append(
                    ("Stress P-norm proxy", self._fmt(data["stress_pnorm"], "MPa"))
                )
            if data.get("volume") is not None:
                topo_rows.append(
                    (
                        "Density-equivalent volume",
                        self._fmt(data["volume"], "mm^3"),
                    )
                )
            if data.get("mass") is not None:
                topo_rows.append(
                    (
                        "Density-equivalent mass",
                        self._fmt(float(data["mass"]) * 1000.0, "kg"),
                    )
                )
            if data.get("recovered_design_volume") is not None:
                topo_rows.append(
                    (
                        "Recovered CAD design volume",
                        self._fmt(data["recovered_design_volume"], "mm^3"),
                    )
                )
            if data.get("recovered_design_mass") is not None:
                topo_rows.append(
                    (
                        "Recovered CAD design mass",
                        self._fmt(
                            float(data["recovered_design_mass"]) * 1000.0,
                            "kg",
                        ),
                    )
                )
            if data.get("recovery_volume_delta_pct") is not None:
                topo_rows.append(
                    (
                        "CAD vs density-volume difference",
                        f"{float(data['recovery_volume_delta_pct']):+.1f}%",
                    )
                )
            if (
                data.get("assembly_hardware_volume") is not None
                and float(data.get("assembly_hardware_volume") or 0.0) > 0.0
            ):
                topo_rows.append(
                    (
                        "Separate joint hardware volume",
                        self._fmt(data["assembly_hardware_volume"], "mm^3"),
                    )
                )
            used_iterations = int(data.get("iterations") or 0)
            max_iterations = int(data.get("max_iterations") or 0)
            iteration_text = (
                f"{used_iterations} of {max_iterations} maximum"
                if max_iterations > 0
                else str(used_iterations)
            )
            topo_rows.append(("Optimizer iterations used", iteration_text))
            topo_rows.append(("Converged", "Yes" if data.get("converged") else "No"))
            if data.get("message"):
                topo_rows.append(("Stop reason", str(data["message"])))
            self._add_section("Topology result", topo_rows)

            validation = data.get("validation_summary")
            if isinstance(validation, dict):
                validation_rows = []
                if validation.get("max_stress") is not None:
                    validation_rows.append(
                        (
                            "Validated peak stress",
                            self._fmt(validation["max_stress"], "MPa"),
                        )
                    )
                if validation.get("compliance") is not None:
                    validation_rows.append(
                        (
                            "Validated compliance",
                            self._fmt(validation["compliance"], "N mm"),
                        )
                    )
                if validation_rows:
                    self._add_section("CalculiX validation", validation_rows)

        # Warnings from the external backends
        warnings = data.get("warnings") or []
        if warnings:
            self._add_warnings(list(warnings))


class LibraryPanel(QtWidgets.QWidget):
    """Component library with categorized nodes."""

    _CATEGORY_ICONS = {
        "Modeling": ("fa5s.cube", "#81C784"),
        "Prepare": ("fa5s.mouse-pointer", "#80CBC4"),
        "FEA": ("fa5s.calculator", "#64B5F6"),
        "Topology": ("fa5s.project-diagram", "#9CCC65"),
        "Crash": ("fa5s.car-crash", "#EF5350"),
        "Measure": ("fa5s.ruler-combined", "#B39DDB"),
        "Data & Export": ("fa5s.file-export", "#90A4AE"),
    }
    _NODE_ICONS = (
        ("com.cad.geometry.box", "fa5s.cube", "#81C784"),
        ("com.cad.geometry.cylinder", "fa5s.database", "#81C784"),
        ("com.cad.geometry.tube", "fa5s.circle", "#81C784"),
        ("com.cad.geometry.cylindrical_shell", "fa5s.circle-notch", "#81C784"),
        ("com.cad.geometry.boolean", "fa5s.object-group", "#81C784"),
        ("com.cad.geometry.through_hole", "fa5s.dot-circle", "#81C784"),
        ("com.cad.geometry.fillet", "fa5s.bezier-curve", "#81C784"),
        ("com.cad.geometry.transform", "fa5s.arrows-alt", "#81C784"),
        ("com.cad.geometry.linear_pattern", "fa5s.grip-horizontal", "#81C784"),
        ("com.cad.freecad_part", "fa5s.drafting-compass", "#81C784"),
        ("com.cad.import_step", "fa5s.file-import", "#90A4AE"),
        ("com.cad.import_stl", "fa5s.file-import", "#90A4AE"),
        ("com.cad.select_face_interactive", "fa5s.mouse-pointer", "#80CBC4"),
        ("com.cad.sim.mesh", "fa5s.project-diagram", "#80CBC4"),
        ("com.cad.sim.material", "fa5s.layer-group", "#64B5F6"),
        ("com.cad.sim.constraint", "fa5s.anchor", "#FF8A65"),
        ("com.cad.sim.load", "fa5s.arrow-down", "#FF8A65"),
        ("com.cad.sim.pressure_load", "fa5s.compress-arrows-alt", "#FF8A65"),
        ("com.cad.sim.solver", "fa5s.play-circle", "#9CCC65"),
        ("com.cad.topopt.support", "fa5s.anchor", "#9CCC65"),
        ("com.cad.topopt.load", "fa5s.arrow-down", "#9CCC65"),
        ("com.cad.topopt.non_design_region", "fa5s.vector-square", "#D6B45C"),
        ("com.cad.topopt.joint", "fa5s.link", "#9CCC65"),
        ("com.cad.topopt.operating_case", "fa5s.sitemap", "#9CCC65"),
        ("com.cad.topopt.thermal_sink", "fa5s.temperature-low", "#9CCC65"),
        ("com.cad.topopt.heat_load", "fa5s.fire", "#9CCC65"),
        ("com.cad.sim.topopt_voxel", "fa5s.project-diagram", "#9CCC65"),
        ("com.cad.sim.remesh", "fa5s.sync-alt", "#9CCC65"),
        ("com.cad.sim.crash_material", "fa5s.shield-alt", "#EF5350"),
        ("com.cad.sim.impact", "fa5s.car-crash", "#EF5350"),
        ("com.cad.sim.crash_solver", "fa5s.play-circle", "#EF5350"),
        ("com.cad.sim.radioss_deck", "fa5s.file-code", "#EF5350"),
        ("com.cad.assembly", "fa5s.cubes", "#B39DDB"),
        ("com.cad.mass_properties", "fa5s.weight", "#B39DDB"),
        ("com.cad.bounding_box", "fa5s.vector-square", "#B39DDB"),
        ("com.cad.measure_distance", "fa5s.ruler", "#B39DDB"),
        ("com.cad.surface_area", "fa5s.draw-polygon", "#B39DDB"),
        ("com.cad.number", "fa5s.hashtag", "#90A4AE"),
        ("com.cad.export_step", "fa5s.file-export", "#90A4AE"),
        ("com.cad.export_stl", "fa5s.file-export", "#90A4AE"),
    )

    @staticmethod
    def _icon_for(label: str):
        try:
            import qtawesome as qta
        except Exception:
            qta = None
        icon_spec = LibraryPanel._CATEGORY_ICONS.get(label)
        if qta is not None and icon_spec is not None:
            try:
                return qta.icon(icon_spec[0], color=icon_spec[1])
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
        return QtWidgets.QApplication.style().standardIcon(
            QtWidgets.QStyle.SP_DirClosedIcon
        )

    @staticmethod
    def _icon_for_node(node_id: str):
        try:
            import qtawesome as qta
        except Exception:
            qta = None
        if qta is not None:
            for prefix, icon_name, color in LibraryPanel._NODE_ICONS:
                if node_id == prefix or node_id.startswith(prefix + "."):
                    try:
                        return qta.icon(icon_name, color=color)
                    except Exception:
                        break
        return QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)

    @staticmethod
    def _calculix_status():
        """Return a concise, non-launching CalculiX availability check."""
        try:
            from pylcss.solver_backends.calculix import resolve_calculix_executable

            executable = resolve_calculix_executable()
        except Exception as exc:
            return False, f"CalculiX availability check failed: {exc}"
        if not executable:
            return False, (
                "CalculiX unavailable. Deck-only generation remains available; "
                "install with `python scripts/install_solvers.py --only ccx`."
            )
        return True, f"CalculiX ready: {executable}"

    @staticmethod
    def _openradioss_status():
        """Return a concise, non-launching availability check for the palette."""
        try:
            from pylcss.solver_backends.execution import resolve_executable
            from pylcss.solver_backends.radioss_reader import resolve_anim_to_vtk

            starter = resolve_executable(
                None,
                ("PYLCSS_OPENRADIOSS_STARTER", "OPENRADIOSS_STARTER"),
                ("starter_win64.exe", "starter_win64", "starter_linux64_gf"),
            )
            engine = resolve_executable(
                None,
                ("PYLCSS_OPENRADIOSS_ENGINE", "OPENRADIOSS_ENGINE"),
                ("engine_win64.exe", "engine_win64", "engine_linux64_gf"),
            )
            converter = resolve_anim_to_vtk()
        except Exception as exc:
            return False, f"Availability check failed: {exc}"

        missing = [
            label
            for label, value in (
                ("Starter", starter),
                ("Engine", engine),
                ("anim_to_vtk", converter),
            )
            if not value
        ]
        if missing:
            return False, (
                "OpenRadioss unavailable: missing " + ", ".join(missing) + ".\n"
                "Deck-only generation remains available; install with "
                "`python scripts/install_solvers.py --only radioss`."
            )
        return True, "OpenRadioss ready: Starter, Engine, and anim_to_vtk detected."

    def __init__(self, spawn_callback):
        super().__init__()
        self.spawn_callback = spawn_callback
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(4)
        radioss_ready, radioss_status = self._openradioss_status()
        calculix_ready, calculix_status = self._calculix_status()

        # Search box — title was redundant with the dock title, dropped.
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search tools")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_tree)
        self.layout.addWidget(self.search)

        # Tree view for categories
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.setIconSize(QtCore.QSize(16, 16))
        self.tree.setUniformRowHeights(True)
        self.tree.setAnimated(True)
        self.tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tree.setTextElideMode(QtCore.Qt.ElideRight)
        self.tree.setStyleSheet(
            """
            QTreeWidget {
                border: none;
                background: transparent;
                outline: none;
            }
            QTreeView::item {
                padding: 4px 2px;
                border: 0;
            }
            QTreeView::item:hover {
                background: rgba(255, 255, 255, 18);
            }
            QTreeView::item:selected {
                background: rgba(33, 150, 243, 60);
                color: white;
            }
            """
        )
        # enable dragging from the library into the graph
        self.tree.setDragEnabled(True)
        self.tree.itemPressed.connect(self._start_drag)

        # Compact public palette. Legacy/script nodes remain registered so old
        # studies load, but they do not duplicate the native interactive tools.
        # Format: (Label, node_id, tooltip_description)
        categories = {
            "Modeling": [
                (
                    "Box",
                    "com.cad.geometry.box",
                    "Create a dimensioned rectangular solid from inspector values.",
                ),
                (
                    "Cylinder",
                    "com.cad.geometry.cylinder",
                    "Create a dimensioned solid cylinder on X, Y, or Z.",
                ),
                (
                    "Tube",
                    "com.cad.geometry.tube",
                    "Create a hollow circular tube from diameter, wall, and length.",
                ),
                (
                    "Cylindrical Shell",
                    "com.cad.geometry.cylindrical_shell",
                    "Create a midsurface tube for shell FEA or crash analysis.",
                ),
                (
                    "Boolean",
                    "com.cad.geometry.boolean",
                    "Union, subtract, or intersect two connected solids.",
                ),
                (
                    "Through Hole",
                    "com.cad.geometry.through_hole",
                    "Cut a dimensioned through-hole without an expression.",
                ),
                (
                    "Fillet",
                    "com.cad.geometry.fillet",
                    "Round all edges or one principal edge family.",
                ),
                (
                    "Transform",
                    "com.cad.geometry.transform",
                    "Translate and rotate a connected solid.",
                ),
                (
                    "Linear Pattern",
                    "com.cad.geometry.linear_pattern",
                    "Create a linear array with optional fusion.",
                ),
                (
                    "FreeCAD Part",
                    "com.cad.freecad_part",
                    "Interactive parametric CAD authored in FreeCAD's own GUI.\n"
                    "Double-click the node to launch FreeCAD on a node-owned .FCStd file.\n"
                    "Draw sketches, add PartDesign features, set named faces / FEM loads;\n"
                    "save inside FreeCAD and PyLCSS auto-imports the geometry via BREP +\n"
                    "sidecar JSON.  Requires FreeCAD installed locally\n"
                    "(`python scripts/install_solvers.py --only freecad`).",
                ),
                (
                    "Import STEP",
                    "com.cad.import_step",
                    "Import a STEP / IGES CAD file as the upstream geometry.",
                ),
                (
                    "Import Mesh",
                    "com.cad.import_stl",
                    "Import an STL / OBJ surface mesh.",
                ),
            ],
            "Prepare": [
                (
                    "Pick Geometry",
                    "com.cad.select_face_interactive",
                    "Click faces, edges, or vertices in the 3D viewer and keep "
                    "the selection with the study.",
                ),
                (
                    "Mesh",
                    "com.cad.sim.mesh",
                    "Create solid or shell elements for FEA and crash analysis.",
                ),
            ],
            "FEA": [
                ("Material", "com.cad.sim.material", "Define the structural material."),
                (
                    "Support",
                    "com.cad.sim.constraint",
                    "Apply a support or prescribed displacement.",
                ),
                ("Force", "com.cad.sim.load", "Apply a resultant force."),
                (
                    "Pressure",
                    "com.cad.sim.pressure_load",
                    "Apply a uniform face pressure.",
                ),
                (
                    "Static Solver",
                    "com.cad.sim.solver",
                    "Run the CalculiX structural solver.",
                ),
            ],
            "Topology": [
                (
                    "Topology Support",
                    "com.cad.topopt.support",
                    "Apply a support to a selected CAD face, edge, or vertex "
                    "without a prebuilt FE mesh.",
                ),
                (
                    "Topology Force",
                    "com.cad.topopt.load",
                    "Apply a resultant force to a selected CAD face, edge, or vertex.",
                ),
                (
                    "Non-Design Region",
                    "com.cad.topopt.non_design_region",
                    "Clamp a connected closed CAD volume as preserved material "
                    "or preserved void.",
                ),
                (
                    "Topology Joint",
                    "com.cad.topopt.joint",
                    "Connect two selected anchor regions with a fixed, revolute, "
                    "spherical, or prismatic kinematic coupling.",
                ),
                (
                    "Operating Case",
                    "com.cad.topopt.operating_case",
                    "Group pose-specific supports, loads, and joints for a "
                    "multi-load or multibody optimization envelope.",
                ),
                (
                    "Temperature Boundary",
                    "com.cad.topopt.thermal_sink",
                    "Hold a selected region at the thermal reference temperature.",
                ),
                (
                    "Heat Input",
                    "com.cad.topopt.heat_load",
                    "Apply total heat input to a selected CAD face, edge, or vertex.",
                ),
                (
                    "Topology Solver",
                    "com.cad.sim.topopt_voxel",
                    "Run structural, thermal, rib, or lattice topology optimization.",
                ),
                (
                    "Volume Remesh",
                    "com.cad.sim.remesh",
                    "Convert a recovered surface into a volume mesh.",
                ),
            ],
            "Crash": [
                (
                    "Crash Material",
                    "com.cad.sim.crash_material",
                    "Elasto-plastic material for an explicit crash study.",
                ),
                (
                    "Impact Setup",
                    "com.cad.sim.impact",
                    "Define the impact scenario, velocity, wall, and contact scope.",
                ),
                (
                    "Crash Solver",
                    "com.cad.sim.crash_solver",
                    "Run the OpenRadioss explicit transient solver.",
                ),
                (
                    "OpenRadioss Deck",
                    "com.cad.sim.radioss_deck",
                    "Run an existing OpenRadioss or LS-DYNA deck.",
                ),
            ],
            "Measure": [
                ("Assembly", "com.cad.assembly", "Combine parts"),
                ("Mass Properties", "com.cad.mass_properties", "Calculate mass/volume"),
                ("Bounding Box", "com.cad.bounding_box", "Measure dimensions"),
                (
                    "Measure Distance",
                    "com.cad.measure_distance",
                    "Measure the minimum distance between two connected CAD shapes.",
                ),
                (
                    "Surface Area",
                    "com.cad.surface_area",
                    "Calculate the surface area of a connected CAD shape.",
                ),
            ],
            "Data & Export": [
                (
                    "Parameter",
                    "com.cad.number",
                    "Reusable numeric input. Give it an Exposed Parameter Name "
                    "to drive it from the Modeling Environment.",
                ),
                (
                    "Export STEP",
                    "com.cad.export_step",
                    "Export the current shape to a .step file",
                ),
                (
                    "Export STL",
                    "com.cad.export_stl",
                    "Export the current mesh / shape to a .stl file",
                ),
            ],
        }

        cat_font = QtGui.QFont()
        cat_font.setBold(True)
        for category, items in categories.items():
            cat_item = QtWidgets.QTreeWidgetItem([category])
            cat_item.setFont(0, cat_font)
            icon = self._icon_for(category)
            if icon is not None:
                cat_item.setIcon(0, icon)
            # Category rows are not draggable / not selectable as a target.
            cat_item.setFlags(cat_item.flags() & ~QtCore.Qt.ItemIsSelectable)

            for item_data in items:
                label, node_id, tooltip = item_data
                item = QtWidgets.QTreeWidgetItem([label])
                item.setData(0, QtCore.Qt.UserRole, node_id)
                item.setIcon(0, self._icon_for_node(node_id))
                if node_id == "com.cad.sim.solver":
                    tooltip += "\n\n" + calculix_status
                    item.setData(0, QtCore.Qt.UserRole + 1, calculix_ready)
                    if not calculix_ready:
                        item.setForeground(0, QtGui.QColor("#FFB74D"))
                elif node_id in (
                    "com.cad.sim.crash_solver",
                    "com.cad.sim.radioss_deck",
                ):
                    tooltip += "\n\n" + radioss_status
                    item.setData(0, QtCore.Qt.UserRole + 1, radioss_ready)
                    if not radioss_ready:
                        item.setForeground(0, QtGui.QColor("#FFB74D"))
                item.setToolTip(0, tooltip)  # Show description on hover
                cat_item.addChild(item)

            self.tree.addTopLevelItem(cat_item)

        # A seven-row category overview fits the default panel. Searching
        # expands only matching groups; clearing the search closes them again.
        self.tree.collapseAll()
        self.tree.itemDoubleClicked.connect(self._on_component_selected)
        self.layout.addWidget(self.tree)

    def _filter_tree(self, text):
        """Filter tree items based on search text."""
        text = text.lower().strip()
        for i in range(self.tree.topLevelItemCount()):
            category = self.tree.topLevelItem(i)
            visible_children = 0
            for j in range(category.childCount()):
                item = category.child(j)
                matches = text in item.text(0).lower() if text else True
                item.setHidden(not matches)
                if matches:
                    visible_children += 1
            # Show category if any children match, or if search is empty
            category.setHidden(visible_children == 0 and bool(text))
            if visible_children > 0 or not text:
                category.setExpanded(bool(text))  # Auto-expand when searching

    def _on_component_selected(self, item, column):
        """Handle component selection from library."""
        node_id = item.data(0, QtCore.Qt.UserRole)
        if node_id:
            self.spawn_callback(node_id, item.text(0))

    def _start_drag(self, item, column):
        """Start a drag operation carrying the node identifier."""
        node_id = item.data(0, QtCore.Qt.UserRole)
        if not node_id:
            return
        drag = QDrag(self.tree)
        mime = QMimeData()
        mime.setData("application/x-node-id", str(node_id).encode("utf-8"))
        mime.setText(item.text(0))
        drag.setMimeData(mime)
        drag.exec(QtCore.Qt.CopyAction)
