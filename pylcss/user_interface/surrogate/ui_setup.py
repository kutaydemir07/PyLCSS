# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SurrogateUiMixin behavior for surrogate training."""

from __future__ import annotations

import logging

import pyqtgraph as pg
import qtawesome as qta
from PySide6 import QtCore, QtWidgets

from pylcss.surrogate_modeling.training_engine import (
    TORCH_AVAILABLE,
)
from pylcss.user_interface.common.theme_manager import COLORS

logger = logging.getLogger(__name__)

__all__ = ["SurrogateUiMixin"]


class SurrogateUiMixin:
    def setup_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        # --- LEFT PANEL: Configuration ---
        config_panel = QtWidgets.QWidget()
        config_panel.setFixedWidth(380)
        config_layout = QtWidgets.QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(config_panel)

        # Sub-tabs: Data / Model / Training -- mirrors the solution-space
        # left-panel pattern so the workflow reads top-to-bottom: pick data,
        # pick model, train.
        self.left_tabs = QtWidgets.QTabWidget()
        config_layout.addWidget(self.left_tabs)

        tab_data = QtWidgets.QWidget()
        data_layout = QtWidgets.QVBoxLayout(tab_data)
        self.left_tabs.addTab(tab_data, "Data")

        tab_model = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(tab_model)
        self.left_tabs.addTab(tab_model, "Model")

        tab_training = QtWidgets.QWidget()
        training_layout = QtWidgets.QVBoxLayout(tab_training)
        self.left_tabs.addTab(tab_training, "Training")

        # 1. Node Selection
        grp_node = QtWidgets.QGroupBox("Target Node")
        l_node = QtWidgets.QVBoxLayout(grp_node)
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
        data_layout.addWidget(grp_node)

        # 2. Data Source
        grp_data = QtWidgets.QGroupBox("Data Source")
        l_data = QtWidgets.QVBoxLayout(grp_data)

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

        self.btn_generate = QtWidgets.QPushButton(" Generate Data")
        self.btn_generate.setIcon(qta.icon("fa5s.magic"))
        self.btn_generate.clicked.connect(self.start_generation)
        l_gen.addRow(self.btn_generate)
        self.stack_data.addWidget(p_gen)

        # Page 2: Upload
        p_upload = QtWidgets.QWidget()
        l_upload = QtWidgets.QVBoxLayout(p_upload)

        self.btn_browse = QtWidgets.QPushButton(" Browse CSV/JSON...")
        self.btn_browse.setIcon(qta.icon("fa5s.folder-open"))
        self.btn_browse.clicked.connect(self.browse_file)
        l_upload.addWidget(self.btn_browse)

        self.lbl_file_info = QtWidgets.QLabel("No file loaded")
        self.lbl_file_info.setStyleSheet("color: gray; font-style: italic;")
        self.lbl_file_info.setWordWrap(True)
        l_upload.addWidget(self.lbl_file_info)
        self.stack_data.addWidget(p_upload)

        l_data.addWidget(self.stack_data)
        data_layout.addWidget(grp_data)
        data_layout.addStretch()

        # 3. Model Architecture
        grp_arch = QtWidgets.QGroupBox("Model Architecture")
        l_arch = QtWidgets.QFormLayout(grp_arch)

        self.combo_algo = QtWidgets.QComboBox()
        model_options = [
            "MLP Regressor",
            "Random Forest",
            "Gradient Boosting",
            "Gaussian Process",
        ]
        if TORCH_AVAILABLE:
            model_options.append("Deep Neural Network (PyTorch)")
        # Geometric backbones require torch + trimesh; check at construction time.
        try:
            from pylcss.surrogate_modeling.geometry import (
                TRIMESH_AVAILABLE as _TRIMESH_OK,
            )
        except ImportError:
            _TRIMESH_OK = False
        if TORCH_AVAILABLE and _TRIMESH_OK:
            model_options.append("Geom-DeepONet")
            model_options.append("GINO")
        self.combo_algo.addItems(model_options)
        self.combo_algo.setToolTip(
            "Choose the machine learning algorithm for the surrogate model:\n"
            "• MLP Regressor: Neural network with configurable layers\n"
            "• Random Forest: Ensemble of decision trees\n"
            "• Gradient Boosting: Sequential tree boosting\n"
            "• Gaussian Process: Probabilistic kernel-based model"
            + ("\n• Deep Neural Network: PyTorch MLP" if TORCH_AVAILABLE else "")
            + (
                "\n• Geom-DeepONet: Geometry-aware operator (CAD per query)"
                if (TORCH_AVAILABLE and _TRIMESH_OK)
                else ""
            )
            + (
                "\n• GINO: FNO on SDF background grid"
                if (TORCH_AVAILABLE and _TRIMESH_OK)
                else ""
            )
        )
        self.combo_algo.currentIndexChanged.connect(self.update_hyperparams)
        l_arch.addRow("Algorithm:", self.combo_algo)

        # Dynamic Hyperparameters Stack
        self.stack_params = QtWidgets.QStackedWidget()

        # --- MLP Params ---
        p_mlp = QtWidgets.QWidget()
        f_mlp = QtWidgets.QFormLayout(p_mlp)

        self.txt_layers = QtWidgets.QLineEdit("(100, 50)")

        self.combo_activ = QtWidgets.QComboBox()
        self.combo_activ.addItems(["relu", "tanh", "logistic", "identity"])
        self.combo_activ.setToolTip(
            "Activation function for neural network layers:\n• relu: Rectified Linear Unit (most common)\n• tanh: Hyperbolic tangent\n• logistic: Sigmoid function\n• identity: Linear activation"
        )

        self.combo_solver = QtWidgets.QComboBox()
        self.combo_solver.addItems(["adam", "lbfgs", "sgd"])
        self.combo_solver.setToolTip(
            "Optimization algorithm for training:\n• adam: Adaptive moment estimation (recommended)\n• lbfgs: Limited-memory BFGS (good for small datasets)\n• sgd: Stochastic gradient descent"
        )

        self.spin_alpha_mlp = QtWidgets.QDoubleSpinBox()
        self.spin_alpha_mlp.setRange(0.00001, 10.0)
        self.spin_alpha_mlp.setValue(0.0001)
        self.spin_alpha_mlp.setDecimals(5)

        self.spin_max_iter = QtWidgets.QSpinBox()
        self.spin_max_iter.setRange(100, 100000)
        self.spin_max_iter.setValue(5000)
        self.spin_max_iter.setSingleStep(500)

        self.chk_early_stopping = QtWidgets.QCheckBox("Early Stopping")
        self.chk_early_stopping.setChecked(False)

        f_mlp.addRow("Hidden Layers:", self.txt_layers)
        f_mlp.addRow("Activation:", self.combo_activ)
        f_mlp.addRow("Solver:", self.combo_solver)
        f_mlp.addRow("Alpha (L2 Penalty):", self.spin_alpha_mlp)
        f_mlp.addRow("Max Iterations:", self.spin_max_iter)
        f_mlp.addRow("", self.chk_early_stopping)
        self.stack_params.addWidget(p_mlp)

        # --- RF Params ---
        p_rf = QtWidgets.QWidget()
        f_rf = QtWidgets.QFormLayout(p_rf)

        self.spin_est_rf = QtWidgets.QSpinBox()
        self.spin_est_rf.setRange(10, 5000)
        self.spin_est_rf.setValue(100)

        self.spin_depth_rf = QtWidgets.QSpinBox()
        self.spin_depth_rf.setRange(0, 1000)
        self.spin_depth_rf.setValue(0)
        self.spin_depth_rf.setSpecialValueText("None")

        self.spin_min_split_rf = QtWidgets.QSpinBox()
        self.spin_min_split_rf.setRange(2, 100)
        self.spin_min_split_rf.setValue(2)

        self.spin_min_leaf_rf = QtWidgets.QSpinBox()
        self.spin_min_leaf_rf.setRange(1, 100)
        self.spin_min_leaf_rf.setValue(1)

        self.chk_bootstrap_rf = QtWidgets.QCheckBox("Bootstrap")
        self.chk_bootstrap_rf.setChecked(True)

        f_rf.addRow("Estimators:", self.spin_est_rf)
        f_rf.addRow("Max Depth:", self.spin_depth_rf)
        f_rf.addRow("Min Samples Split:", self.spin_min_split_rf)
        f_rf.addRow("Min Samples Leaf:", self.spin_min_leaf_rf)
        f_rf.addRow("", self.chk_bootstrap_rf)
        self.stack_params.addWidget(p_rf)

        # --- GB Params ---
        p_gb = QtWidgets.QWidget()
        f_gb = QtWidgets.QFormLayout(p_gb)

        self.spin_est_gb = QtWidgets.QSpinBox()
        self.spin_est_gb.setRange(10, 5000)
        self.spin_est_gb.setValue(100)

        self.spin_lr_gb = QtWidgets.QDoubleSpinBox()
        self.spin_lr_gb.setRange(0.001, 1.0)
        self.spin_lr_gb.setValue(0.1)
        self.spin_lr_gb.setSingleStep(0.01)

        self.spin_depth_gb = QtWidgets.QSpinBox()
        self.spin_depth_gb.setRange(1, 100)
        self.spin_depth_gb.setValue(3)

        self.spin_subsample_gb = QtWidgets.QDoubleSpinBox()
        self.spin_subsample_gb.setRange(0.1, 1.0)
        self.spin_subsample_gb.setValue(1.0)
        self.spin_subsample_gb.setSingleStep(0.1)

        self.combo_loss_gb = QtWidgets.QComboBox()
        self.combo_loss_gb.addItems(
            ["squared_error", "absolute_error", "huber", "quantile"]
        )
        self.combo_loss_gb.setToolTip(
            "Loss function for gradient boosting:\n• squared_error: Mean squared error\n• absolute_error: Mean absolute error\n• huber: Huber loss (robust to outliers)\n• quantile: Quantile regression"
        )

        f_gb.addRow("Estimators:", self.spin_est_gb)
        f_gb.addRow("Learning Rate:", self.spin_lr_gb)
        f_gb.addRow("Max Depth:", self.spin_depth_gb)
        f_gb.addRow("Subsample:", self.spin_subsample_gb)
        f_gb.addRow("Loss Function:", self.combo_loss_gb)
        self.stack_params.addWidget(p_gb)

        # --- GP Params ---
        p_gp = QtWidgets.QWidget()
        f_gp = QtWidgets.QFormLayout(p_gp)

        self.spin_alpha_gp = QtWidgets.QDoubleSpinBox()
        self.spin_alpha_gp.setRange(1e-10, 1e-1)
        self.spin_alpha_gp.setValue(1e-6)
        self.spin_alpha_gp.setSingleStep(1e-7)
        self.spin_alpha_gp.setDecimals(10)

        self.spin_restarts_gp = QtWidgets.QSpinBox()
        self.spin_restarts_gp.setRange(0, 100)
        self.spin_restarts_gp.setValue(15)

        self.chk_normalize_gp = QtWidgets.QCheckBox("Normalize Y")
        self.chk_normalize_gp.setChecked(True)

        f_gp.addRow("Alpha (noise):", self.spin_alpha_gp)
        f_gp.addRow("Optimizer Restarts:", self.spin_restarts_gp)
        f_gp.addRow("", self.chk_normalize_gp)
        self.stack_params.addWidget(p_gp)

        # --- PyTorch Params ---
        p_pytorch = QtWidgets.QWidget()
        f_pytorch = QtWidgets.QFormLayout(p_pytorch)

        self.spin_lr_pytorch = QtWidgets.QDoubleSpinBox()
        self.spin_lr_pytorch.setRange(1e-6, 1.0)
        self.spin_lr_pytorch.setValue(0.01)
        self.spin_lr_pytorch.setSingleStep(0.001)
        self.spin_lr_pytorch.setDecimals(6)
        f_pytorch.addRow("Learning Rate:", self.spin_lr_pytorch)

        self.spin_batch_size = QtWidgets.QSpinBox()
        self.spin_batch_size.setRange(8, 2048)
        self.spin_batch_size.setValue(32)
        self.spin_batch_size.setSingleStep(8)
        f_pytorch.addRow("Batch Size:", self.spin_batch_size)

        self.txt_hidden_layers = QtWidgets.QLineEdit("64, 64")
        f_pytorch.addRow("Hidden Layers:", self.txt_hidden_layers)

        self.combo_optimizer = QtWidgets.QComboBox()
        self.combo_optimizer.addItems(["Adam", "SGD", "RMSprop", "Adagrad"])
        self.combo_optimizer.setToolTip(
            "PyTorch optimizer algorithm:\n• Adam: Adaptive moment estimation (recommended)\n• SGD: Stochastic gradient descent\n• RMSprop: Root mean square propagation\n• Adagrad: Adaptive gradient algorithm"
        )
        f_pytorch.addRow("Optimizer:", self.combo_optimizer)

        self.combo_pt_activation = QtWidgets.QComboBox()
        self.combo_pt_activation.addItems(["ReLU", "Tanh", "Sigmoid", "LeakyReLU"])
        self.combo_pt_activation.setToolTip(
            "Activation function for PyTorch neural network:\n• ReLU: Rectified Linear Unit (most common)\n• Tanh: Hyperbolic tangent\n• Sigmoid: Logistic function\n• LeakyReLU: Leaky version of ReLU"
        )
        f_pytorch.addRow("Activation:", self.combo_pt_activation)

        self.spin_pt_dropout = QtWidgets.QDoubleSpinBox()
        self.spin_pt_dropout.setRange(0.0, 0.9)
        self.spin_pt_dropout.setValue(0.1)  # Default to 0.1 for uncertainty estimation
        self.spin_pt_dropout.setSingleStep(0.1)
        f_pytorch.addRow("Dropout Rate:", self.spin_pt_dropout)

        self.spin_epochs = QtWidgets.QSpinBox()
        self.spin_epochs.setRange(10, 100000)
        self.spin_epochs.setValue(2000)
        self.spin_epochs.setSingleStep(500)
        f_pytorch.addRow("Epochs:", self.spin_epochs)

        self.spin_mc_samples = QtWidgets.QSpinBox()
        self.spin_mc_samples.setRange(10, 1000)
        self.spin_mc_samples.setValue(50)
        self.spin_mc_samples.setSingleStep(10)
        self.spin_mc_samples.setToolTip(
            "Number of Monte Carlo samples for uncertainty quantification.\nHigher values give more accurate uncertainty estimates but take longer."
        )
        f_pytorch.addRow("MC Samples:", self.spin_mc_samples)

        self.stack_params.addWidget(p_pytorch)

        # --- Geom-DeepONet / GINO shared params (geometric backbones) ---
        # Both backbones need: CAD graph path + solver kind + nodal field name +
        # sample count + epochs + a small set of architecture knobs.
        p_geom = QtWidgets.QWidget()
        f_geom = QtWidgets.QFormLayout(p_geom)

        cad_row = QtWidgets.QHBoxLayout()
        self.txt_cad_path = QtWidgets.QLineEdit()
        self.txt_cad_path.setPlaceholderText("Path to .cad / .json CAD graph...")
        self.txt_cad_path.setToolTip(
            "PyLCSS CAD graph file. The wrapper will call pylcss.design_studio.runtime "
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
        self.combo_cad_kind.addItems(["fea", "crash", "topopt"])
        self.combo_cad_kind.setToolTip(
            "Which CAD terminal solver to call (fea/crash/topopt)."
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
        # Map algorithm display name -> stack page index. Used by
        # update_hyperparams to switch the hyperparam panel.
        self._algo_to_page = {
            "MLP Regressor": 0,
            "Random Forest": 1,
            "Gradient Boosting": 2,
            "Gaussian Process": 3,
            "Deep Neural Network (PyTorch)": 4,
            "Geom-DeepONet": self._geom_page_index,
            "GINO": self._geom_page_index,
        }

        l_arch.addRow(self.stack_params)

        # Make the architecture section scrollable
        scroll_arch = QtWidgets.QScrollArea()
        scroll_arch.setWidget(grp_arch)
        scroll_arch.setWidgetResizable(True)
        scroll_arch.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_arch.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_arch.setMaximumHeight(400)  # Limit height to make it manageable
        # Make background transparent to match theme
        scroll_arch.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        scroll_arch.setAutoFillBackground(False)

        model_layout.addWidget(scroll_arch)
        model_layout.addStretch()

        # 3.5. Training Mode
        grp_mode = QtWidgets.QGroupBox("Training Mode")
        l_mode = QtWidgets.QVBoxLayout(grp_mode)
        self.radio_standard = QtWidgets.QRadioButton("Standard")
        self.radio_debug = QtWidgets.QRadioButton("Debug (Sanity Check)")
        self.radio_debug.setToolTip(
            "WARNING: Debug mode trains and tests on the same data, guaranteeing perfect scores.\nThis is ONLY for sanity checking - NEVER use debug models for real engineering design!"
        )
        self.radio_standard.setChecked(True)
        self.radio_standard.toggled.connect(self.toggle_debug_mode)
        self.radio_debug.toggled.connect(self.toggle_debug_mode)
        l_mode.addWidget(self.radio_standard)
        l_mode.addWidget(self.radio_debug)
        training_layout.addWidget(grp_mode)

        # Debug Mode Warning Label
        self.lbl_debug_warning = QtWidgets.QLabel(
            "⚠️ VALIDATION DISABLED - DO NOT USE FOR DESIGN ⚠️"
        )
        self.lbl_debug_warning.setStyleSheet("""
            QLabel {
                background-color: #ff4444;
                color: white;
                font-weight: bold;
                font-size: 12pt;
                padding: 8px;
                border-radius: 4px;
                border: 2px solid #cc0000;
            }
        """)
        self.lbl_debug_warning.setVisible(False)
        self.lbl_debug_warning.setWordWrap(True)
        training_layout.addWidget(self.lbl_debug_warning)

        # Debug Buttons (initially hidden)
        self.btn_overfit1 = QtWidgets.QPushButton("Overfit 1 Sample")
        self.btn_overfit1.clicked.connect(lambda: self.start_debug_training(1))
        self.btn_overfit1.setVisible(False)
        training_layout.addWidget(self.btn_overfit1)

        self.btn_overfit10 = QtWidgets.QPushButton("Overfit 10 Samples")
        self.btn_overfit10.clicked.connect(lambda: self.start_debug_training(10))
        self.btn_overfit10.setVisible(False)
        training_layout.addWidget(self.btn_overfit10)

        # 4. Action Buttons
        self.btn_train = QtWidgets.QPushButton(qta.icon("fa5s.cogs"), " Train Model")
        self.btn_train.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_train.setToolTip(
            "Train the chosen machine learning algorithm using the available data."
        )
        self.btn_train.clicked.connect(self.start_training)
        self.btn_train.setEnabled(False)  # Disabled until data is ready
        training_layout.addWidget(self.btn_train)

        self.btn_save = QtWidgets.QPushButton(
            qta.icon("fa5s.save"), " Save and Attach to Node"
        )
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_model)
        training_layout.addWidget(self.btn_save)

        self.btn_adaptive = QtWidgets.QPushButton(
            " Adaptive Training (Active Learning)"
        )
        self.btn_adaptive.setToolTip(
            "Train surrogate model using adaptive sampling to focus on high-uncertainty regions"
        )
        self.btn_adaptive.clicked.connect(self.start_adaptive_training)
        self.btn_adaptive.setEnabled(False)  # Disabled until data is ready
        training_layout.addWidget(self.btn_adaptive)

        training_layout.addStretch()

        # --- RIGHT PANEL: Visualization ---
        viz_panel = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(viz_panel)

        # Metrics
        self.lbl_metrics = QtWidgets.QLabel("Status: Ready to train.")
        self.lbl_metrics.setStyleSheet(
            "background-color: #333; color: #fff; padding: 10px; border-radius: 4px;"
        )
        viz_layout.addWidget(self.lbl_metrics)

        # Tabs for plots
        self.tab_widget = QtWidgets.QTabWidget()

        # Tab 1: Learning Curves
        self.curve_tab = QtWidgets.QWidget()
        curve_layout = QtWidgets.QVBoxLayout(self.curve_tab)
        self.curve_plot = pg.PlotWidget(title="Learning Curves")
        self.curve_plot.setBackground(COLORS["chart_bg"])
        self.curve_plot.setLabel("left", "Loss (MSE)")
        self.curve_plot.setLabel("bottom", "Epoch/Iteration")
        self.curve_plot.showGrid(x=True, y=True)
        self.curve_plot.addLegend()
        self.train_curve = self.curve_plot.plot(
            pen=pg.mkPen("r", width=2), name="Train Loss"
        )
        self.val_curve = self.curve_plot.plot(
            pen=pg.mkPen("g", width=2), name="Val Loss"
        )
        # Add text item for progress messages
        self.progress_text = pg.TextItem(
            "", anchor=(0.5, 0.5), color=COLORS["chart_fg"]
        )
        self.progress_text.setPos(0, 0)  # Center of data coordinates
        self.curve_plot.addItem(self.progress_text)
        curve_layout.addWidget(self.curve_plot)
        self.tab_widget.addTab(self.curve_tab, "Learning Curves")
        self.tab_widget.setTabToolTip(
            0,
            "Training loss curves (MSE) - Real-time for PyTorch, post-training for MLP Regressor",
        )

        # Tab 2: Parity Plot
        self.parity_tab = QtWidgets.QWidget()
        parity_layout = QtWidgets.QVBoxLayout(self.parity_tab)
        self.plot_widget = pg.PlotWidget(title="Parity Plot (Predicted vs Actual)")
        self.plot_widget.setBackground(COLORS["chart_bg"])
        self.plot_widget.setLabel("left", "Predicted")
        self.plot_widget.setLabel("bottom", "Actual")
        self.plot_widget.showGrid(x=True, y=True)
        parity_layout.addWidget(self.plot_widget)
        self.tab_widget.addTab(self.parity_tab, "Parity Plot")

        # Tab 3: Data Preview
        self.data_tab = QtWidgets.QWidget()
        data_layout = QtWidgets.QVBoxLayout(self.data_tab)
        self.data_table = QtWidgets.QTableWidget()
        data_layout.addWidget(self.data_table)
        self.tab_widget.addTab(self.data_tab, "Data Preview")

        # Tab 4: Cross-Validation Results
        self.cv_tab = QtWidgets.QWidget()
        cv_layout = QtWidgets.QVBoxLayout(self.cv_tab)
        self.cv_table = QtWidgets.QTableWidget()
        self.cv_table.setColumnCount(5)
        self.cv_table.setHorizontalHeaderLabels(
            ["Model", "R² Mean", "R² Std", "RMSE Mean", "MAE Mean"]
        )
        self.cv_table.horizontalHeader().setStretchLastSection(True)
        cv_layout.addWidget(self.cv_table)

        cv_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run_cv = QtWidgets.QPushButton(
            qta.icon("fa5s.crosshairs"), " Run Cross-Validation"
        )
        self.btn_run_cv.setToolTip(
            "Evaluate current model with K-Fold cross-validation"
        )
        self.btn_run_cv.clicked.connect(self._run_cross_validation)
        self.btn_run_cv.setEnabled(False)
        cv_btn_layout.addWidget(self.btn_run_cv)

        self.btn_compare = QtWidgets.QPushButton(
            qta.icon("fa5s.balance-scale"), " Compare All Models"
        )
        self.btn_compare.setToolTip(
            "Compare all available algorithms on the current dataset"
        )
        self.btn_compare.clicked.connect(self._compare_models)
        self.btn_compare.setEnabled(False)
        cv_btn_layout.addWidget(self.btn_compare)

        self.spin_cv_folds = QtWidgets.QSpinBox()
        self.spin_cv_folds.setRange(2, 20)
        self.spin_cv_folds.setValue(5)
        self.spin_cv_folds.setPrefix("Folds: ")
        cv_btn_layout.addWidget(self.spin_cv_folds)

        cv_layout.addLayout(cv_btn_layout)
        self.tab_widget.addTab(self.cv_tab, "Cross-Validation")

        # Tab 5: Feature Importance
        self.fi_tab = QtWidgets.QWidget()
        fi_layout = QtWidgets.QVBoxLayout(self.fi_tab)
        self.fi_plot = pg.PlotWidget(title="Feature Importance")
        self.fi_plot.setBackground(COLORS["chart_bg"])
        self.fi_plot.setLabel("left", "Importance")
        self.fi_plot.setLabel("bottom", "Feature")
        self.fi_plot.showGrid(x=True, y=True, alpha=0.3)
        fi_layout.addWidget(self.fi_plot)

        self.btn_feature_imp = QtWidgets.QPushButton(
            qta.icon("fa5s.sort-amount-down"), " Compute Feature Importance"
        )
        self.btn_feature_imp.setToolTip("Permutation-based feature importance analysis")
        self.btn_feature_imp.clicked.connect(self._compute_feature_importance)
        self.btn_feature_imp.setEnabled(False)
        fi_layout.addWidget(self.btn_feature_imp)
        self.tab_widget.addTab(self.fi_tab, "Feature Importance")

        viz_layout.addWidget(self.tab_widget)

        # Progress and Stop
        progress_layout = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.btn_stop = QtWidgets.QPushButton("Stop Training")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        progress_layout.addWidget(self.progress)
        progress_layout.addWidget(self.btn_stop)
        viz_layout.addLayout(progress_layout)

        layout.addWidget(config_panel)
        layout.addWidget(viz_panel)
