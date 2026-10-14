# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SurrogateUiMixin behavior for surrogate training."""

from __future__ import annotations

import logging

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from pylcss.mdao.surrogate_modeling.training_engine import (
    TORCH_AVAILABLE,
)
from pylcss.user_interface.common.theme_manager import COLORS
from pylcss.user_interface.common.ui_kit import CollapsibleSection as _CollapsibleSection

logger = logging.getLogger(__name__)

__all__ = ["SurrogateUiMixin"]


class _StopButton(QtWidgets.QPushButton):
    """Stop button that reports enabled/text changes so the progress row can follow."""

    state_changed = QtCore.Signal()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.EnabledChange:
            self.state_changed.emit()

    def setText(self, text: str) -> None:
        super().setText(text)
        self.state_changed.emit()


class _EmptyStateOverlay(QtCore.QObject):
    """Centered hint shown over a plot until it holds data."""

    def __init__(self, plot_widget: pg.PlotWidget, text: str) -> None:
        super().__init__(plot_widget)
        self._plot = plot_widget
        self._label = QtWidgets.QLabel(text, plot_widget)
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._label.setStyleSheet(
            "color: gray; font-size: 11pt; background: transparent;"
        )
        plot_widget.installEventFilter(self)
        # pyqtgraph has no single signal for clear()/plot(), so poll cheaply.
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Resize:
            self._label.setGeometry(self._plot.rect())
        return False

    def refresh(self) -> None:
        has_data = False
        for item in self._plot.listDataItems():
            x, _ = item.getData()
            if x is not None and len(x) > 0:
                has_data = True
                break
        self._label.setGeometry(self._plot.rect())
        self._label.setVisible(not has_data)
        if not has_data:
            self._label.raise_()


class SurrogateUiMixin:
    def setup_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Main horizontal splitter between configuration and visualization
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.main_splitter)

        # --- LEFT PANEL: Configuration ---
        config_panel = QtWidgets.QWidget()
        config_panel.setMinimumWidth(380)
        config_layout = QtWidgets.QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(4)
        self.main_splitter.addWidget(config_panel)

        # One scrollable column of collapsible steps: target -> data -> model -> training.
        scroll_steps = QtWidgets.QScrollArea()
        scroll_steps.setWidgetResizable(True)
        scroll_steps.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_steps.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_steps.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_steps.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll_steps.setAutoFillBackground(False)

        steps_content = QtWidgets.QWidget()
        steps_layout = QtWidgets.QVBoxLayout(steps_content)
        steps_layout.setContentsMargins(6, 6, 6, 6)
        steps_layout.setSpacing(8)
        scroll_steps.setWidget(steps_content)
        config_layout.addWidget(scroll_steps, 1)

        self.sec_target = _CollapsibleSection("1  Target node")
        self.sec_data = _CollapsibleSection("2  Data")
        self.sec_model = _CollapsibleSection("3  Model")
        self.sec_train = _CollapsibleSection("4  Training options")
        for _sec in (self.sec_target, self.sec_data, self.sec_model, self.sec_train):
            steps_layout.addWidget(_sec)
        steps_layout.addStretch()
        self.sec_train.set_expanded(False)

        target_layout = self.sec_target.body_layout
        data_layout = self.sec_data.body_layout
        model_layout = self.sec_model.body_layout
        training_layout = self.sec_train.body_layout

        # 1. Node Selection
        grp_node = QtWidgets.QWidget()
        l_node = QtWidgets.QVBoxLayout(grp_node)
        l_node.setContentsMargins(0, 0, 0, 0)
        self.combo_nodes = QtWidgets.QComboBox()
        self.combo_nodes.setToolTip(
            "Select the target node (output variable) to create a surrogate model for. The surrogate will predict this output based on the input variables."
        )
        self.btn_refresh = QtWidgets.QPushButton("Refresh Node List")
        self.btn_refresh.setToolTip(
            "Refresh the list of available nodes from the current modeling environment."
        )
        self.btn_refresh.clicked.connect(self.refresh_nodes)
        l_node.addWidget(self.combo_nodes)
        l_node.addWidget(self.btn_refresh)
        target_layout.addWidget(grp_node)

        # 2. Data Source
        grp_data = QtWidgets.QWidget()
        l_data = QtWidgets.QVBoxLayout(grp_data)
        l_data.setContentsMargins(0, 0, 0, 0)

        # Source Selection
        src_layout = QtWidgets.QHBoxLayout()
        self.radio_gen = QtWidgets.QRadioButton("Generate from Model")
        self.radio_upload = QtWidgets.QRadioButton("Upload File")
        self.radio_gen.setChecked(True)
        self.radio_gen.toggled.connect(self.toggle_data_source)
        src_layout.addWidget(self.radio_gen)
        src_layout.addWidget(self.radio_upload)
        l_data.addLayout(src_layout)

        # Stack for options
        self.stack_data = QtWidgets.QStackedWidget()

        # Page 1: Generation
        p_gen = QtWidgets.QWidget()
        l_gen = QtWidgets.QFormLayout(p_gen)
        self.spin_samples = QtWidgets.QSpinBox()
        self.spin_samples.setRange(100, 1000000)
        self.spin_samples.setValue(1000)
        self.spin_samples.setSingleStep(500)
        l_gen.addRow("Sample Count:", self.spin_samples)

        self.btn_generate = QtWidgets.QPushButton("Generate Data")
        self.btn_generate.clicked.connect(self.start_generation)
        l_gen.addRow(self.btn_generate)
        self.stack_data.addWidget(p_gen)

        # Page 2: Upload
        p_upload = QtWidgets.QWidget()
        l_upload = QtWidgets.QVBoxLayout(p_upload)

        self.btn_browse = QtWidgets.QPushButton("Browse CSV/JSON...")
        self.btn_browse.clicked.connect(self.browse_file)
        l_upload.addWidget(self.btn_browse)

        self.lbl_file_info = QtWidgets.QLabel("No file loaded")
        self.lbl_file_info.setStyleSheet("color: gray; font-style: italic;")
        self.lbl_file_info.setWordWrap(True)
        l_upload.addWidget(self.lbl_file_info)
        self.stack_data.addWidget(p_upload)

        l_data.addWidget(self.stack_data)
        data_layout.addWidget(grp_data)

        # 3. Model Architecture
        grp_arch = QtWidgets.QWidget()
        l_arch = QtWidgets.QFormLayout(grp_arch)
        l_arch.setContentsMargins(0, 0, 0, 0)

        self.combo_algo = QtWidgets.QComboBox()
        model_options = [
            "Auto Mode",
        ]
        if TORCH_AVAILABLE:
            model_options.append("Your Own Architecture (Deep Learning)")
        self.combo_algo.addItems(model_options)
        self.combo_algo.setToolTip(
            "Choose surrogate modeling workflow:\n"
            "- Auto Mode: Automatic multi-model benchmarking and champion selection across peer-reviewed MDAO algorithms (Kriging, KPLS, Polyharmonic RBF, Sparse PCE, SVR, Goel-Haftka Ensembles)\n"
            "- Your Own Architecture (Deep Learning): Visual or PyTorch deep learning architecture builder (Residual blocks, Fourier embeddings, LayerNorm, MC Dropout)"
        )
        self.combo_algo.currentIndexChanged.connect(self.update_hyperparams)
        l_arch.addRow("Algorithm:", self.combo_algo)

        # Dynamic Hyperparameters Stack
        self.stack_params = QtWidgets.QStackedWidget()

        # --- Geom-DeepONet / GINO shared params (geometric backbones) ---
        # Both backbones need: CAD graph path + solver kind + nodal field name +
        # sample count + epochs + a small set of architecture knobs.
        p_geom = QtWidgets.QWidget()
        f_geom = QtWidgets.QFormLayout(p_geom)

        cad_row = QtWidgets.QHBoxLayout()
        self.txt_cad_path = QtWidgets.QLineEdit()
        self.txt_cad_path.setPlaceholderText("Path to .cad / .json CAD graph...")
        self.txt_cad_path.setToolTip(
            "PyLCSS CAD graph file. The wrapper will call pylcss.cae.design_studio.runtime "
            "with each design's parameters to materialise its mesh."
        )
        self.btn_browse_cad = QtWidgets.QPushButton("Browse...")
        self.btn_browse_cad.clicked.connect(self._browse_cad_path)
        cad_row.addWidget(self.txt_cad_path, 1)
        cad_row.addWidget(self.btn_browse_cad)
        cad_holder = QtWidgets.QWidget()
        cad_holder.setLayout(cad_row)
        f_geom.addRow("CAD Graph:", cad_holder)

        self.combo_cad_kind = QtWidgets.QComboBox()
        self.combo_cad_kind.addItems(["fea", "impact", "topopt"])
        self.combo_cad_kind.setToolTip(
            "Which CAD terminal solver to call (fea/impact/topopt)."
        )
        self.combo_cad_kind.currentTextChanged.connect(self._refresh_field_choices)
        f_geom.addRow("Solver:", self.combo_cad_kind)

        # Field is editable in case the user has a custom raw key, but the
        # dropdown is pre-populated with the names PyLCSS's known backends
        # actually emit -- so the common case is a single click.
        self.combo_field = QtWidgets.QComboBox()
        self.combo_field.setEditable(True)
        self.combo_field.setToolTip(
            "Which nodal field the surrogate will predict.\n"
            "Choices are filtered by the selected solver."
        )
        f_geom.addRow("Field:", self.combo_field)
        self._refresh_field_choices(self.combo_cad_kind.currentText())

        self.spin_geom_samples = QtWidgets.QSpinBox()
        self.spin_geom_samples.setRange(4, 1000)
        self.spin_geom_samples.setValue(30)
        self.spin_geom_samples.setToolTip(
            "Number of LHS-sampled design points used to train (one CAD eval each)."
        )
        f_geom.addRow("CAD Samples:", self.spin_geom_samples)

        self.spin_geom_epochs = QtWidgets.QSpinBox()
        self.spin_geom_epochs.setRange(10, 5000)
        self.spin_geom_epochs.setValue(300)
        f_geom.addRow("Epochs:", self.spin_geom_epochs)

        self.spin_geom_lr = QtWidgets.QDoubleSpinBox()
        self.spin_geom_lr.setRange(1e-5, 1.0)
        self.spin_geom_lr.setDecimals(5)
        self.spin_geom_lr.setValue(1e-3)
        self.spin_geom_lr.setSingleStep(1e-4)
        f_geom.addRow("Learning rate:", self.spin_geom_lr)

        # ---- Geom-DeepONet-specific knobs (branch/trunk MLP). ----
        self.spin_donet_latent = QtWidgets.QSpinBox()
        self.spin_donet_latent.setRange(8, 512)
        self.spin_donet_latent.setValue(64)
        self.spin_donet_latent.setToolTip(
            "Inner-product dimension between branch (params) and trunk (geometry) outputs."
        )
        self._lbl_donet_latent = QtWidgets.QLabel("Latent dim:")
        f_geom.addRow(self._lbl_donet_latent, self.spin_donet_latent)

        self.spin_donet_trunk = QtWidgets.QSpinBox()
        self.spin_donet_trunk.setRange(16, 512)
        self.spin_donet_trunk.setValue(64)
        self.spin_donet_trunk.setToolTip("Hidden width of the SIREN trunk network.")
        self._lbl_donet_trunk = QtWidgets.QLabel("Trunk hidden:")
        f_geom.addRow(self._lbl_donet_trunk, self.spin_donet_trunk)

        # ---- GINO-specific knobs (3-D FNO on background SDF grid). ----
        self.spin_gino_channels = QtWidgets.QSpinBox()
        self.spin_gino_channels.setRange(4, 256)
        self.spin_gino_channels.setValue(16)
        self.spin_gino_channels.setToolTip("Width of the FNO hidden channels.")
        self._lbl_gino_channels = QtWidgets.QLabel("Hidden channels:")
        f_geom.addRow(self._lbl_gino_channels, self.spin_gino_channels)

        self.spin_gino_grid = QtWidgets.QSpinBox()
        self.spin_gino_grid.setRange(8, 64)
        self.spin_gino_grid.setValue(24)
        self.spin_gino_grid.setToolTip(
            "Background-grid resolution per axis (R x R x R SDF volume)."
        )
        self._lbl_gino_grid = QtWidgets.QLabel("Grid resolution:")
        f_geom.addRow(self._lbl_gino_grid, self.spin_gino_grid)

        self.spin_gino_modes = QtWidgets.QSpinBox()
        self.spin_gino_modes.setRange(2, 16)
        self.spin_gino_modes.setValue(4)
        self.spin_gino_modes.setToolTip(
            "Number of low-frequency Fourier modes mixed by each FNO layer (per axis)."
        )
        self._lbl_gino_modes = QtWidgets.QLabel("FNO modes:")
        f_geom.addRow(self._lbl_gino_modes, self.spin_gino_modes)

        # Single page (geometric backbones share CAD/training config); the
        # backbone-specific arch rows above get hidden/shown by
        # _set_geom_visibility() when the user picks Geom-DeepONet vs GINO.
        # Default-hide both groups so only the active backbone's knobs show
        # the moment the user lands on the page.
        for _lbl, _w in [
            (self._lbl_donet_latent, self.spin_donet_latent),
            (self._lbl_donet_trunk, self.spin_donet_trunk),
            (self._lbl_gino_channels, self.spin_gino_channels),
            (self._lbl_gino_grid, self.spin_gino_grid),
            (self._lbl_gino_modes, self.spin_gino_modes),
        ]:
            _lbl.setVisible(False)
            _w.setVisible(False)
        self._geom_page_index = self.stack_params.addWidget(p_geom)

        # --- Auto Mode Params ---
        p_auto = QtWidgets.QWidget()
        f_auto = QtWidgets.QFormLayout(p_auto)
        f_auto.setContentsMargins(4, 6, 4, 6)
        f_auto.setSpacing(8)

        lbl_auto_info = QtWidgets.QLabel(
            "Auto Mode benchmarks peer-reviewed MDAO surrogate architectures "
            "(Kriging with ARD, KPLS, Polyharmonic RBF, Sparse Legendre PCE, "
            "Anisotropic SVR, and Goel-Haftka Convex Ensembles) to identify the optimal surrogate."
        )
        lbl_auto_info.setWordWrap(True)
        lbl_auto_info.setStyleSheet("color: #b0b0b0; font-size: 8.5pt; padding-bottom: 4px;")
        f_auto.addRow(lbl_auto_info)

        self.combo_auto_preset = QtWidgets.QComboBox()
        self.combo_auto_preset.addItem("Quick (Kriging, RBF, PCE, SVR)", "quick")
        self.combo_auto_preset.addItem("Standard (+ KPLS, Matern 3/2)", "standard")
        self.combo_auto_preset.addItem("Thorough (Full search & Ensembling)", "thorough")
        self.combo_auto_preset.setToolTip(
            "Search Presets:\n"
            "- Quick: Fast evaluation of 4 core models (Kriging, RBF, PCE, SVR)\n"
            "- Standard: Adds high-dimensional KPLS and Matern 3/2 rough response\n"
            "- Thorough: Adds degree-3 PCE, Gaussian Kriging, and full Goel-Haftka ensemble"
        )
        f_auto.addRow("Search Preset:", self.combo_auto_preset)

        self.combo_auto_metric = QtWidgets.QComboBox()
        self.combo_auto_metric.addItem("R2 Score (Maximize)", "R2")
        self.combo_auto_metric.addItem("RMSE (Minimize)", "RMSE")
        self.combo_auto_metric.addItem("MAE (Minimize)", "MAE")
        self.combo_auto_metric.addItem("RMAE (Constraint Safety)", "RMAE")
        self.combo_auto_metric.setToolTip(
            "Optimization Metric for Champion Selection:\n"
            "- R2: Maximize coefficient of determination (overall fit)\n"
            "- RMSE: Minimize root mean squared error (penalizes large residuals)\n"
            "- MAE: Minimize mean absolute error (robust against outliers)\n"
            "- RMAE: Minimize relative maximum absolute error (critical for constraint safety)"
        )
        f_auto.addRow("Primary Metric:", self.combo_auto_metric)

        self.chk_auto_ensemble = QtWidgets.QCheckBox("Enable Goel-Haftka Optimal Ensemble")
        self.chk_auto_ensemble.setToolTip(
            "Solves quadratic programming min error covariance across top candidate models to create an optimal convex ensemble."
        )
        self.chk_auto_ensemble.setChecked(True)
        f_auto.addRow("", self.chk_auto_ensemble)

        self._auto_page_index = self.stack_params.addWidget(p_auto)

        # --- Custom Architecture Params ---
        p_custom = QtWidgets.QWidget()
        l_custom = QtWidgets.QVBoxLayout(p_custom)
        l_custom.setContentsMargins(0, 0, 0, 0)

        from .architecture_builder_widget import ArchitectureBuilderWidget

        self.arch_builder = ArchitectureBuilderWidget()
        l_custom.addWidget(self.arch_builder)

        self._custom_page_index = self.stack_params.addWidget(p_custom)

        # Map algorithm display name -> stack page index. Used by
        # update_hyperparams to switch the hyperparam panel.
        self._algo_to_page = {
            "Auto Mode": self._auto_page_index,
            "Auto Mode (Recommended)": self._auto_page_index,
            "Auto Mode": self._auto_page_index,
            "Your Own Architecture (Deep Learning)": self._custom_page_index,
            "Custom Architecture (Deep Learning)": self._custom_page_index,
            "Your Own Architecture": self._custom_page_index,
            "Custom Architecture": self._custom_page_index,
            "Geom-DeepONet": self._geom_page_index,
            "GINO": self._geom_page_index,
        }
        self.update_hyperparams(self.combo_algo.currentIndex())

        l_arch.addRow(self.stack_params)
        model_layout.addWidget(grp_arch)

        # 4. Action Buttons (pinned below the tabs so they stay visible on every tab)
        self.lbl_train_hint = QtWidgets.QLabel()
        self.lbl_train_hint.setWordWrap(True)
        self.lbl_train_hint.setStyleSheet("color: gray; font-style: italic; padding: 0 6px;")
        config_layout.addWidget(self.lbl_train_hint)

        self.btn_train = QtWidgets.QPushButton("Train Model")
        self.btn_train.setStyleSheet("font-weight: bold; font-size: 10pt; padding: 10px;")
        self.btn_train.setToolTip(
            "Train the surrogate model using the configured architecture and training dataset."
        )
        self.btn_train.clicked.connect(self.start_training)
        self.btn_train.setEnabled(False)  # Disabled until data is ready
        config_layout.addWidget(self.btn_train)

        self.btn_save = QtWidgets.QPushButton("Save and Attach to Node")
        self.btn_save.setStyleSheet("padding: 8px;")
        self.btn_save.setToolTip(
            "Persist the trained surrogate model and connect it to the target system node."
        )
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_model)
        config_layout.addWidget(self.btn_save)

        # 5. Active-learning configuration (Collapsible Advanced Section)
        self.grp_active = QtWidgets.QGroupBox("Adaptive Sampling (Active Learning)")
        self.grp_active.setCheckable(True)
        self.grp_active.setChecked(False)
        self.grp_active.setToolTip(
            "Optional: Iterative sampling to query new simulation points where surrogate uncertainty is highest."
        )
        l_active_grp = QtWidgets.QVBoxLayout(self.grp_active)
        active_margins = l_active_grp.contentsMargins()

        self.widget_active_content = QtWidgets.QWidget()
        l_active = QtWidgets.QFormLayout(self.widget_active_content)
        self._al_random_state = 42
        self._al_gp_restarts = 3

        self.combo_al_strategy = QtWidgets.QComboBox()
        self.combo_al_strategy.addItem("Committee (GP-RF)", "committee")
        self.combo_al_strategy.addItem("Uncertainty", "uncertainty")
        self.combo_al_strategy.addItem("Random (control)", "random")
        self.combo_al_strategy.setToolTip(
            "Committee combines GP uncertainty with GP-RF disagreement."
        )
        l_active.addRow("Strategy:", self.combo_al_strategy)

        self.spin_al_rounds = QtWidgets.QSpinBox()
        self.spin_al_rounds.setRange(1, 100)
        self.spin_al_rounds.setValue(5)
        l_active.addRow("Rounds:", self.spin_al_rounds)

        self.spin_al_batch = QtWidgets.QSpinBox()
        self.spin_al_batch.setRange(1, 1000)
        self.spin_al_batch.setValue(10)
        l_active.addRow("Batch size:", self.spin_al_batch)

        self.spin_al_candidates = QtWidgets.QSpinBox()
        self.spin_al_candidates.setRange(10, 100000)
        self.spin_al_candidates.setSingleStep(100)
        self.spin_al_candidates.setValue(1000)
        l_active.addRow("Candidate pool:", self.spin_al_candidates)

        self.spin_al_explore = QtWidgets.QDoubleSpinBox()
        self.spin_al_explore.setRange(0.0, 1.0)
        self.spin_al_explore.setDecimals(2)
        self.spin_al_explore.setSingleStep(0.05)
        self.spin_al_explore.setValue(0.30)
        l_active.addRow("Explore floor:", self.spin_al_explore)

        self.spin_al_min_dist = QtWidgets.QDoubleSpinBox()
        self.spin_al_min_dist.setRange(0.0, 1.0)
        self.spin_al_min_dist.setDecimals(3)
        self.spin_al_min_dist.setSingleStep(0.01)
        self.spin_al_min_dist.setValue(0.06)
        l_active.addRow("Minimum distance:", self.spin_al_min_dist)

        self.lbl_al_budget = QtWidgets.QLabel()
        l_active.addRow("New simulations:", self.lbl_al_budget)
        self.spin_al_rounds.valueChanged.connect(self._update_active_budget_label)
        self.spin_al_batch.valueChanged.connect(self._update_active_budget_label)
        self._update_active_budget_label()

        self.btn_adaptive = QtWidgets.QPushButton("Run Adaptive Training")
        self.btn_adaptive.setToolTip(
            "Iteratively execute simulations at points of maximum model uncertainty."
        )
        self.btn_adaptive.clicked.connect(self.start_adaptive_training)
        self.btn_adaptive.setEnabled(False)  # Disabled until data is ready
        l_active.addRow(self.btn_adaptive)

        l_active_grp.addWidget(self.widget_active_content)
        def _toggle_active(on: bool) -> None:
            self.widget_active_content.setVisible(on)
            # Collapse the empty body so an unchecked group is just its title.
            if on:
                l_active_grp.setContentsMargins(active_margins)
            else:
                l_active_grp.setContentsMargins(0, 0, 0, 0)

        _toggle_active(False)
        self.grp_active.toggled.connect(_toggle_active)

        training_layout.addWidget(self.grp_active)

        # --- RIGHT PANEL: Visualization ---
        viz_panel = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(viz_panel)

        # Metrics
        self.lbl_metrics = QtWidgets.QLabel(
            "No dataset yet. Generate or upload data to begin."
        )
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setStyleSheet(
            "background-color: #1e1f22; border: 1px solid #282a2e; color: #b5bac1; padding: 10px; border-radius: 4px;"
        )
        viz_layout.addWidget(self.lbl_metrics)

        # Tabs for plots
        self.tab_widget = QtWidgets.QTabWidget()

        # Tab 1: Parity Plot (Universal validation for all algorithms)
        self.parity_tab = QtWidgets.QWidget()
        parity_layout = QtWidgets.QVBoxLayout(self.parity_tab)
        self.plot_widget = pg.PlotWidget(title="Parity Plot (Predicted vs Actual)")
        self.plot_widget.setBackground(COLORS["chart_bg"])
        self.plot_widget.setLabel("left", "Predicted")
        self.plot_widget.setLabel("bottom", "Actual")
        self.plot_widget.showGrid(x=True, y=True)
        parity_layout.addWidget(self.plot_widget)
        self._parity_empty = _EmptyStateOverlay(
            self.plot_widget,
            "No results yet.\nGenerate data, then press Train Model.",
        )
        self.tab_widget.addTab(self.parity_tab, "Parity Plot")
        self.tab_widget.setTabToolTip(
            0,
            "Predictions and test values.",
        )

        # Tab 2: AutoML Leaderboard (Multi-model tournament results)
        self.lb_tab = QtWidgets.QWidget()
        lb_layout = QtWidgets.QVBoxLayout(self.lb_tab)
        self.lb_table = QtWidgets.QTableWidget()
        self.lb_table.setColumnCount(6)
        self.lb_table.setHorizontalHeaderLabels(
            ["Rank", "Model", "R²", "RMSE", "RMAE", "Fit Time (s)"]
        )
        lb_header = self.lb_table.horizontalHeader()
        lb_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        lb_header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        lb_layout.addWidget(self.lb_table)
        self.tab_widget.addTab(self.lb_tab, "Leaderboard")
        self.tab_widget.setTabToolTip(
            1,
            "Model scores and training time.",
        )

        # Tab 3: Training Curves (Loss vs Epoch for iterative neural networks)
        self.curve_tab = QtWidgets.QWidget()
        curve_layout = QtWidgets.QVBoxLayout(self.curve_tab)
        self.curve_plot = pg.PlotWidget(title="Training Curves (Epoch vs MSE Loss)")
        self.curve_plot.setBackground(COLORS["chart_bg"])
        self.curve_plot.setLabel("left", "Loss (MSE)")
        self.curve_plot.setLabel("bottom", "Epoch")
        self.curve_plot.showGrid(x=True, y=True)
        self.curve_plot.addLegend()
        self.train_curve = self.curve_plot.plot(
            pen=pg.mkPen("r", width=2), name="Train Loss"
        )
        self.val_curve = self.curve_plot.plot(
            pen=pg.mkPen("g", width=2), name="Val Loss"
        )
        self.progress_text = pg.TextItem(
            "", anchor=(0.5, 0.5), color=COLORS["chart_fg"]
        )
        self.progress_text.setPos(0, 0)
        self.curve_plot.addItem(self.progress_text)
        curve_layout.addWidget(self.curve_plot)
        self._curve_empty = _EmptyStateOverlay(
            self.curve_plot,
            "Loss curves appear here while a neural architecture trains.",
        )
        self.tab_widget.addTab(self.curve_tab, "Training Curves")
        self.tab_widget.setTabToolTip(
            2,
            "Training and validation loss.",
        )

        viz_layout.addWidget(self.tab_widget)

        # Progress and Stop
        self.progress_row = QtWidgets.QWidget()
        progress_layout = QtWidgets.QHBoxLayout(self.progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress = QtWidgets.QProgressBar()
        self.btn_stop = _StopButton("Stop Training")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        progress_layout.addWidget(self.progress)
        progress_layout.addWidget(self.btn_stop)
        viz_layout.addWidget(self.progress_row)
        self.btn_stop.state_changed.connect(self._sync_progress_row)
        self.progress.valueChanged.connect(self._sync_progress_row)
        self._sync_progress_row()

        self.main_splitter.addWidget(viz_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([450, 750])

        # Keep the step headers (state mark + summary) in sync with the workflow
        self.combo_nodes.currentIndexChanged.connect(self.update_training_summary)
        self.combo_algo.currentIndexChanged.connect(self.update_training_summary)
        self.grp_active.toggled.connect(self.update_training_summary)
        self.spin_al_rounds.valueChanged.connect(self.update_training_summary)
        self.update_training_summary()

    def _sync_progress_row(self) -> None:
        """Show progress and Stop only while a task is running."""
        stop = self.btn_stop
        running = stop.isEnabled() or stop.text() == "Stopping..."
        busy = 0 < self.progress.value() < 100
        self.progress_row.setVisible(running or busy)
        stop.setVisible(running)

    def update_training_summary(self) -> None:
        """Refresh step summaries and state marks; expand the next step when one completes."""
        if not hasattr(self, "sec_train"):
            return
        node_txt = self.combo_nodes.currentText()
        algo_txt = self.combo_algo.currentText()
        X_train = getattr(self, "X_train", None)
        has_data = X_train is not None

        if has_data:
            X_test = getattr(self, "X_test", None)
            n_total = len(X_train) + (len(X_test) if X_test is not None else 0)
            shape = getattr(X_train, "shape", ())
            n_inputs = shape[1] if len(shape) > 1 else 1
            data_txt = f"{n_total} samples, {n_inputs} inputs"
        else:
            data_txt = "No dataset loaded"

        trained = getattr(self, "current_model", None) is not None
        if trained:
            train_txt = "Model trained"
        elif self.grp_active.isChecked():
            train_txt = f"Adaptive sampling, {self.spin_al_rounds.value()} rounds"
        else:
            train_txt = "Standard training"

        steps = (
            ("target", self.sec_target, bool(node_txt), node_txt or "No node selected"),
            ("data", self.sec_data, has_data, data_txt),
            ("model", self.sec_model, bool(algo_txt), algo_txt or "None selected"),
            ("train", self.sec_train, trained, train_txt),
        )
        previous = getattr(self, "_step_states", {})
        states = {}
        current_assigned = False
        for key, section, done, text in steps:
            if done:
                state = "done"
            elif not current_assigned:
                state = "current"
                current_assigned = True
            else:
                state = "todo"
            states[key] = state
            section.set_state(state)
            section.set_summary(text)
            if key == "train":
                continue  # optional step stays collapsed unless the user opens it
            if state != previous.get(key):
                if state == "done":
                    section.set_expanded(False)
                elif state == "current":
                    section.set_expanded(True)
        self._step_states = states

        if hasattr(self, "lbl_train_hint"):
            self.lbl_train_hint.setText(
                "" if has_data else "Generate or upload data in step 2 to enable training."
            )
            self.lbl_train_hint.setVisible(not has_data)
