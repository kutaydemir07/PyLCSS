# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
User interface for Surrogate Model Training.
Provides advanced configuration, training visualization, and model management.
"""

from typing import Optional, List, Dict, Any, Tuple
from PySide6 import QtWidgets, QtCore
import logging
import numpy as np
import joblib
import pyqtgraph as pg
import qtawesome as qta
from .training_engine import SurrogateTrainer, SKLEARN_AVAILABLE, TORCH_AVAILABLE
from ..system_modeling.model_builder import GraphBuilder
import os
import time 

logger = logging.getLogger(__name__)

class SurrogateTrainingWidget(QtWidgets.QWidget):
    """
    Main widget for the Surrogate Model Training tab.
    """
    
    def __init__(self, modeling_widget: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__()
        self.modeling_widget: Optional[QtWidgets.QWidget] = modeling_widget
        self.current_model: Optional[Any] = None
        self.current_metrics: Optional[Dict[str, Any]] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.epochs: List[int] = []
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.setup_ui()
        
        if not SKLEARN_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "Missing Dependency", 
                "Scikit-learn is required for this feature.\nPlease install it: pip install scikit-learn")
            self.setEnabled(False)

    def setup_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        
        # --- LEFT PANEL: Configuration ---
        config_panel = QtWidgets.QWidget()
        config_panel.setFixedWidth(380)
        config_layout = QtWidgets.QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(config_panel)
        
        # 1. Node Selection
        grp_node = QtWidgets.QGroupBox("Target Node")
        l_node = QtWidgets.QVBoxLayout(grp_node)
        self.combo_nodes = QtWidgets.QComboBox()
        self.combo_nodes.setToolTip("Select the target node (output variable) to create a surrogate model for. The surrogate will predict this output based on the input variables.")
        self.btn_refresh = QtWidgets.QPushButton("Refresh Node List")
        self.btn_refresh.setToolTip("Refresh the list of available nodes from the current modeling environment.")
        self.btn_refresh.clicked.connect(self.refresh_nodes)
        l_node.addWidget(self.combo_nodes)
        l_node.addWidget(self.btn_refresh)
        config_layout.addWidget(grp_node)
        
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
        self.btn_generate.setIcon(qta.icon('fa5s.magic'))
        self.btn_generate.clicked.connect(self.start_generation)
        l_gen.addRow(self.btn_generate)
        self.stack_data.addWidget(p_gen)
        
        # Page 2: Upload
        p_upload = QtWidgets.QWidget()
        l_upload = QtWidgets.QVBoxLayout(p_upload)
        
        self.btn_browse = QtWidgets.QPushButton(" Browse CSV/JSON...")
        self.btn_browse.setIcon(qta.icon('fa5s.folder-open'))
        self.btn_browse.clicked.connect(self.browse_file)
        l_upload.addWidget(self.btn_browse)
        
        self.lbl_file_info = QtWidgets.QLabel("No file loaded")
        self.lbl_file_info.setStyleSheet("color: gray; font-style: italic;")
        self.lbl_file_info.setWordWrap(True)
        l_upload.addWidget(self.lbl_file_info)
        self.stack_data.addWidget(p_upload)
        
        l_data.addWidget(self.stack_data)
        config_layout.addWidget(grp_data)
        
        # 3. Model Architecture
        grp_arch = QtWidgets.QGroupBox("Model Architecture")
        l_arch = QtWidgets.QFormLayout(grp_arch)
        
        self.combo_algo = QtWidgets.QComboBox()
        model_options = ["MLP Regressor", "Random Forest", "Gradient Boosting", "Gaussian Process"]
        if TORCH_AVAILABLE:
            model_options.append("Deep Neural Network (PyTorch)")
        self.combo_algo.addItems(model_options)
        self.combo_algo.setToolTip("Choose the machine learning algorithm for the surrogate model:\n• MLP Regressor: Neural network with configurable layers\n• Random Forest: Ensemble of decision trees\n• Gradient Boosting: Sequential tree boosting\n• Gaussian Process: Probabilistic kernel-based model" + ("\n• Deep Neural Network: Advanced PyTorch-based neural network" if TORCH_AVAILABLE else "\n• Deep Neural Network: Requires PyTorch (not available)"))
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
        self.combo_activ.setToolTip("Activation function for neural network layers:\n• relu: Rectified Linear Unit (most common)\n• tanh: Hyperbolic tangent\n• logistic: Sigmoid function\n• identity: Linear activation")
        
        self.combo_solver = QtWidgets.QComboBox()
        self.combo_solver.addItems(["adam", "lbfgs", "sgd"])
        self.combo_solver.setToolTip("Optimization algorithm for training:\n• adam: Adaptive moment estimation (recommended)\n• lbfgs: Limited-memory BFGS (good for small datasets)\n• sgd: Stochastic gradient descent")
        
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
        self.spin_est_rf.setRange(10, 5000); self.spin_est_rf.setValue(100)
        
        self.spin_depth_rf = QtWidgets.QSpinBox()
        self.spin_depth_rf.setRange(0, 1000); self.spin_depth_rf.setValue(0)
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
        self.spin_est_gb.setRange(10, 5000); self.spin_est_gb.setValue(100)
        
        self.spin_lr_gb = QtWidgets.QDoubleSpinBox()
        self.spin_lr_gb.setRange(0.001, 1.0); self.spin_lr_gb.setValue(0.1); self.spin_lr_gb.setSingleStep(0.01)
        
        self.spin_depth_gb = QtWidgets.QSpinBox()
        self.spin_depth_gb.setRange(1, 100)
        self.spin_depth_gb.setValue(3)
        
        self.spin_subsample_gb = QtWidgets.QDoubleSpinBox()
        self.spin_subsample_gb.setRange(0.1, 1.0)
        self.spin_subsample_gb.setValue(1.0)
        self.spin_subsample_gb.setSingleStep(0.1)
        
        self.combo_loss_gb = QtWidgets.QComboBox()
        self.combo_loss_gb.addItems(["squared_error", "absolute_error", "huber", "quantile"])
        self.combo_loss_gb.setToolTip("Loss function for gradient boosting:\n• squared_error: Mean squared error\n• absolute_error: Mean absolute error\n• huber: Huber loss (robust to outliers)\n• quantile: Quantile regression")
        
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
        self.spin_alpha_gp.setRange(1e-10, 1e-1); self.spin_alpha_gp.setValue(1e-6); self.spin_alpha_gp.setSingleStep(1e-7)
        self.spin_alpha_gp.setDecimals(10)
        
        self.spin_restarts_gp = QtWidgets.QSpinBox()
        self.spin_restarts_gp.setRange(0, 100); self.spin_restarts_gp.setValue(15)
        
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
        self.combo_optimizer.setToolTip("PyTorch optimizer algorithm:\n• Adam: Adaptive moment estimation (recommended)\n• SGD: Stochastic gradient descent\n• RMSprop: Root mean square propagation\n• Adagrad: Adaptive gradient algorithm")
        f_pytorch.addRow("Optimizer:", self.combo_optimizer)
        
        self.combo_pt_activation = QtWidgets.QComboBox()
        self.combo_pt_activation.addItems(["ReLU", "Tanh", "Sigmoid", "LeakyReLU"])
        self.combo_pt_activation.setToolTip("Activation function for PyTorch neural network:\n• ReLU: Rectified Linear Unit (most common)\n• Tanh: Hyperbolic tangent\n• Sigmoid: Logistic function\n• LeakyReLU: Leaky version of ReLU")
        f_pytorch.addRow("Activation:", self.combo_pt_activation)
        
        self.spin_pt_dropout = QtWidgets.QDoubleSpinBox()
        self.spin_pt_dropout.setRange(0.0, 0.9)
        self.spin_pt_dropout.setValue(0.0)
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
        self.spin_mc_samples.setToolTip("Number of Monte Carlo samples for uncertainty quantification.\nHigher values give more accurate uncertainty estimates but take longer.")
        f_pytorch.addRow("MC Samples:", self.spin_mc_samples)
        
        self.stack_params.addWidget(p_pytorch)
        
        l_arch.addRow(self.stack_params)
        config_layout.addWidget(grp_arch)
        
        # 3.5. Training Mode
        grp_mode = QtWidgets.QGroupBox("Training Mode")
        l_mode = QtWidgets.QVBoxLayout(grp_mode)
        self.radio_standard = QtWidgets.QRadioButton("Standard")
        self.radio_debug = QtWidgets.QRadioButton("Debug (Sanity Check)")
        self.radio_debug.setToolTip("WARNING: Debug mode trains and tests on the same data, guaranteeing perfect scores.\nThis is ONLY for sanity checking - NEVER use debug models for real engineering design!")
        self.radio_standard.setChecked(True)
        self.radio_standard.toggled.connect(self.toggle_debug_mode)
        self.radio_debug.toggled.connect(self.toggle_debug_mode)
        l_mode.addWidget(self.radio_standard)
        l_mode.addWidget(self.radio_debug)
        config_layout.addWidget(grp_mode)
        
        # Debug Mode Warning Label
        self.lbl_debug_warning = QtWidgets.QLabel("⚠️ VALIDATION DISABLED - DO NOT USE FOR DESIGN ⚠️")
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
        config_layout.addWidget(self.lbl_debug_warning)
        
        # Debug Buttons (initially hidden)
        self.btn_overfit1 = QtWidgets.QPushButton("Overfit 1 Sample")
        self.btn_overfit1.clicked.connect(lambda: self.start_debug_training(1))
        self.btn_overfit1.setVisible(False)
        config_layout.addWidget(self.btn_overfit1)
        
        self.btn_overfit10 = QtWidgets.QPushButton("Overfit 10 Samples")
        self.btn_overfit10.clicked.connect(lambda: self.start_debug_training(10))
        self.btn_overfit10.setVisible(False)
        config_layout.addWidget(self.btn_overfit10)
        
        # 4. Action Buttons
        self.btn_train = QtWidgets.QPushButton(qta.icon('fa5s.cogs'), " Train Model")
        self.btn_train.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_train.setToolTip("Train the chosen machine learning algorithm using the available data.")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_train.setEnabled(False) # Disabled until data is ready
        config_layout.addWidget(self.btn_train)
        
        self.btn_save = QtWidgets.QPushButton(qta.icon('fa5s.save'), " Save and Attach to Node")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_model)
        config_layout.addWidget(self.btn_save)
        
        config_layout.addStretch()
        
        # --- RIGHT PANEL: Visualization ---
        viz_panel = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(viz_panel)
        
        # Metrics
        self.lbl_metrics = QtWidgets.QLabel("Status: Ready to train.")
        self.lbl_metrics.setStyleSheet("background-color: #333; color: #fff; padding: 10px; border-radius: 4px;")
        viz_layout.addWidget(self.lbl_metrics)
        
        # Tabs for plots
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Tab 1: Learning Curves
        self.curve_tab = QtWidgets.QWidget()
        curve_layout = QtWidgets.QVBoxLayout(self.curve_tab)
        self.curve_plot = pg.PlotWidget(title="Learning Curves")
        self.curve_plot.setBackground('w')  # White background
        self.curve_plot.setLabel('left', 'Loss (MSE)')
        self.curve_plot.setLabel('bottom', 'Epoch/Iteration')
        self.curve_plot.showGrid(x=True, y=True)
        self.curve_plot.addLegend()
        self.train_curve = self.curve_plot.plot(pen=pg.mkPen('r', width=2), name='Train Loss')
        self.val_curve = self.curve_plot.plot(pen=pg.mkPen('g', width=2), name='Val Loss')
        # Add text item for progress messages
        self.progress_text = pg.TextItem("", anchor=(0.5, 0.5), color=(255, 255, 255))
        self.progress_text.setPos(0, 0)  # Center of data coordinates
        self.curve_plot.addItem(self.progress_text)
        curve_layout.addWidget(self.curve_plot)
        self.tab_widget.addTab(self.curve_tab, "Learning Curves")
        self.tab_widget.setTabToolTip(0, "Training loss curves (MSE) - Real-time for PyTorch, post-training for MLP Regressor")
        
        # Tab 2: Parity Plot
        self.parity_tab = QtWidgets.QWidget()
        parity_layout = QtWidgets.QVBoxLayout(self.parity_tab)
        self.plot_widget = pg.PlotWidget(title="Parity Plot (Predicted vs Actual)")
        self.plot_widget.setBackground('w')  # White background
        self.plot_widget.setLabel('left', 'Predicted')
        self.plot_widget.setLabel('bottom', 'Actual')
        self.plot_widget.showGrid(x=True, y=True)
        parity_layout.addWidget(self.plot_widget)
        self.tab_widget.addTab(self.parity_tab, "Parity Plot")

        # Tab 3: Data Preview
        self.data_tab = QtWidgets.QWidget()
        data_layout = QtWidgets.QVBoxLayout(self.data_tab)
        self.data_table = QtWidgets.QTableWidget()
        data_layout.addWidget(self.data_table)
        self.tab_widget.addTab(self.data_tab, "Data Preview")
        
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

    def toggle_data_source(self):
        if self.radio_gen.isChecked():
            self.stack_data.setCurrentIndex(0)
        else:
            self.stack_data.setCurrentIndex(1)

    def browse_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Data File", "", "Data Files (*.csv *.json)")
        if not fname:
            return
            
        try:
            import pandas as pd
            if fname.endswith('.json'):
                df = pd.read_json(fname)
            else:
                df = pd.read_csv(fname)
                
            # We need to identify inputs and outputs
            # For now, we'll try to match with the selected node's inputs and output
            idx = self.combo_nodes.currentIndex()
            if idx < 0:
                QtWidgets.QMessageBox.warning(self, "Error", "Please select a target node first to match columns.")
                return
                
            target_node = self.combo_nodes.itemData(idx)
            
            # Get input names from the graph connection logic or node properties
            # This is tricky without the full graph context easily available in a simple way
            # Let's try to get inputs from the node's input ports
            # But wait, the surrogate models the WHOLE subgraph feeding into this node?
            # Or just this node? 
            # Usually surrogate models replace a complex calculation. 
            # If it's a "Black Box" node, it has inputs.
            # If it's an output node of a graph, it depends on system inputs.
            
            # Let's assume the user knows what they are doing and the CSV has columns:
            # inputs... and output
            
            # Heuristic: 
            # 1. Find column matching target node name (or 'y', 'target', 'output')
            # 2. All other numeric columns are inputs
            
            target_col = None
            possible_targets = [target_node.name(), target_node.get_property('var_name'), 'y', 'target', 'output']
            
            for t in possible_targets:
                if t and t in df.columns:
                    target_col = t
                    break
            
            if not target_col:
                # Ask user to pick target column?
                cols = list(df.columns)
                item, ok = QtWidgets.QInputDialog.getItem(self, "Select Target Column", 
                                                        "Which column is the output?", cols, 0, False)
                if ok and item:
                    target_col = item
                else:
                    return

            # Prepare data
            y = df[target_col].values
            X_df = df.drop(columns=[target_col])
            
            # Filter only numeric columns for X
            X_df = X_df.select_dtypes(include=[np.number])
            X = X_df.values
            
            # Split
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.lbl_file_info.setText(f"Loaded: {os.path.basename(fname)}\n{len(df)} samples\n{X.shape[1]} inputs")
            self.btn_train.setEnabled(True)
            self.lbl_metrics.setText("Data loaded. Ready to train.")
            self.update_data_table()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))

    def start_generation(self):
        idx = self.combo_nodes.currentIndex()
        if idx < 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No node selected.")
            return
            
        target_node = self.combo_nodes.itemData(idx)
        samples = self.spin_samples.value()
        
        self.btn_generate.setEnabled(False)
        self.lbl_metrics.setText("Generating data...")
        self.progress.setValue(0)
        
        try:
            graph = self.modeling_widget.current_graph
            nodes = graph.all_nodes()
            input_nodes = [n for n in nodes if n.type_.startswith('com.pfd.input')]
            output_nodes = [n for n in nodes if n.type_.startswith('com.pfd.output')]
            
            builder = GraphBuilder(graph)
            spy_code, spy_inputs, spy_outputs = builder.build_spy_model(
                nodes, input_nodes, output_nodes, target_node.id, "spy_model"
            )
            
            input_bounds = []
            for inp_node in input_nodes:
                if inp_node.has_property('input_props'):
                    props = inp_node.get_property('input_props')
                    min_val = float(props.get('min', '0.0'))
                    max_val = float(props.get('max', '10.0'))
                else:
                    min_val = float(inp_node.get_property('min'))
                    max_val = float(inp_node.get_property('max'))
                input_bounds.append((min_val, max_val))
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Preparation Error", str(e))
            self.btn_generate.setEnabled(True)
            return

        self.gen_worker = DataGenerationWorker(
            spy_code, spy_inputs, spy_outputs, input_bounds, samples
        )
        self.gen_worker.progress_sig.connect(self.update_progress)
        self.gen_worker.done_sig.connect(self.generation_finished)
        self.gen_worker.start()

    def generation_finished(self, data, error):
        self.btn_generate.setEnabled(True)
        if error:
            QtWidgets.QMessageBox.critical(self, "Generation Failed", error)
            self.lbl_metrics.setText("Generation failed.")
            return
            
        self.X_train, self.y_train, self.X_test, self.y_test = data
        self.btn_train.setEnabled(True)
        self.lbl_metrics.setText(f"Data generated: {len(self.X_train) + len(self.X_test)} samples.")
        self.progress.setValue(100)
        self.update_data_table()

    def update_data_table(self):
        if self.X_train is None:
            return
            
        # Combine train and test for preview
        X = np.vstack((self.X_train, self.X_test))
        y = np.concatenate((self.y_train, self.y_test))
        
        # Handle y shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        rows, x_cols = X.shape
        _, y_cols = y.shape
        
        self.data_table.setRowCount(min(rows, 1000)) # Limit to 1000 rows for performance
        self.data_table.setColumnCount(x_cols + y_cols)
        
        headers = [f"Input {i+1}" for i in range(x_cols)] + [f"Output {i+1}" if y_cols > 1 else "Output" for i in range(y_cols)]
        self.data_table.setHorizontalHeaderLabels(headers)
        
        for i in range(min(rows, 1000)):
            for j in range(x_cols):
                self.data_table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{X[i, j]:.4f}"))
            for k in range(y_cols):
                val = y[i, k]
                self.data_table.setItem(i, x_cols + k, QtWidgets.QTableWidgetItem(f"{val:.4f}"))

    def refresh_nodes(self) -> None:
        """Fetch available CustomBlockNodes from the active graph."""
        self.combo_nodes.clear()
        if not self.modeling_widget or not self.modeling_widget.current_graph:
            return
            
        nodes = self.modeling_widget.current_graph.all_nodes()
        for node in nodes:
            if node.type_.startswith('com.pfd.custom_block'):
                # Store node ID in user data
                self.combo_nodes.addItem(f"{node.name()} ({node.id})", node)

    def update_hyperparams(self, index: int) -> None:
        self.stack_params.setCurrentIndex(index)

    def toggle_debug_mode(self) -> None:
        is_debug = self.radio_debug.isChecked()
        self.btn_overfit1.setVisible(is_debug)
        self.btn_overfit10.setVisible(is_debug)
        self.btn_train.setVisible(not is_debug)
        self.lbl_debug_warning.setVisible(is_debug)

    def start_debug_training(self, num_samples: int) -> None:
        idx = self.combo_nodes.currentIndex()
        if idx < 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No node selected.")
            return
            
        import os
        target_node = self.combo_nodes.itemData(idx)
        config = self.get_config()
        config['debug_mode'] = True
        config['num_samples'] = num_samples
        config['validation_split'] = 0.0  # No validation
        config['epochs'] = 10000  # High epochs for overfitting
        
        self.btn_overfit1.setEnabled(False)
        self.btn_overfit10.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.curve_plot.clear()
        self.plot_widget.clear()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
        # Recreate curves after clearing
        self.train_curve = self.curve_plot.plot(pen=pg.mkPen('r', width=2), name='Train Loss')
        self.val_curve = self.curve_plot.plot(pen=pg.mkPen('g', width=2), name='Val Loss')
        
        self.tab_widget.setCurrentWidget(self.curve_tab)
        
        # Show training message for debug mode (always PyTorch-like behavior)
        self.progress_text.setText("Debug Training...\n(Overfitting test)")
        self.progress_text.show()
        # Set plot range to show the text
        self.curve_plot.setXRange(-1, 1)
        self.curve_plot.setYRange(-1, 1)
        
        # Same preparation as start_training
        try:
            graph = self.modeling_widget.current_graph
            nodes = graph.all_nodes()
            input_nodes = [n for n in nodes if n.type_.startswith('com.pfd.input')]
            output_nodes = [n for n in nodes if n.type_.startswith('com.pfd.output')]
            
            builder = GraphBuilder(graph)
            spy_code, spy_inputs, spy_outputs = builder.build_spy_model(
                nodes, input_nodes, output_nodes, target_node.id, "spy_model"
            )
            
            input_bounds = []
            for inp_node in input_nodes:
                if inp_node.has_property('input_props'):
                    props = inp_node.get_property('input_props')
                    min_val = float(props.get('min', '0.0'))
                    max_val = float(props.get('max', '10.0'))
                else:
                    min_val = float(inp_node.get_property('min'))
                    max_val = float(inp_node.get_property('max'))
                input_bounds.append((min_val, max_val))
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Preparation Error", str(e))
            self.btn_overfit1.setEnabled(True)
            self.btn_overfit10.setEnabled(True)
            return

        self.worker = TrainingWorker(
            spy_code, spy_inputs, spy_outputs, input_bounds,
            num_samples, config
        )
        self.worker.progress_sig.connect(self.update_progress)
        self.worker.loss_sig.connect(self.update_loss_plot)
        self.worker.done_sig.connect(self.training_finished)
        self.worker.start()
        self.btn_stop.setEnabled(True)

    def stop_training(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop_flag = True
            self.btn_stop.setText("Stopping...")
            self.btn_stop.setEnabled(False)

    def update_loss_plot(self, data):
        epoch = data['epoch']
        train_loss = data['train']
        val_loss = data['val']
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Hide progress text when we start getting real data
        self.progress_text.hide()
        
        # Throttle GUI updates to prevent freezing during fast training
        current_time = time.time()
        if not hasattr(self, '_last_plot_time'):
            self._last_plot_time = 0
        
        if current_time - self._last_plot_time > 0.1:  # 10 FPS limit
            # Check if items are actually in the plot item list
            plot_items = self.curve_plot.listDataItems()
            
            if not hasattr(self, 'train_curve') or self.train_curve not in plot_items:
                self.train_curve = self.curve_plot.plot(pen=pg.mkPen('r', width=2), name='Train Loss')
            
            if not hasattr(self, 'val_curve') or self.val_curve not in plot_items:
                self.val_curve = self.curve_plot.plot(pen=pg.mkPen('g', width=2), name='Val Loss')
            
            # Safe updates
            self.train_curve.setData(np.array(self.epochs), np.array(self.train_losses))
            self.val_curve.setData(np.array(self.epochs), np.array(self.val_losses))
            self.curve_plot.update()
            self.curve_plot.autoRange()
            self._last_plot_time = current_time

    def get_config(self) -> Dict[str, Any]:
        """Gather configuration from UI."""
        algo = self.combo_algo.currentText()
        config = {'model_type': algo}
        
        if algo == "MLP Regressor":
            config['hidden_layers'] = self.txt_layers.text()
            config['activation'] = self.combo_activ.currentText()
            config['solver'] = self.combo_solver.currentText()
            config['alpha'] = self.spin_alpha_mlp.value()
            config['max_iter'] = self.spin_max_iter.value()
            config['early_stopping'] = self.chk_early_stopping.isChecked()
            
        elif algo == "Random Forest":
            config['n_estimators'] = self.spin_est_rf.value()
            d = self.spin_depth_rf.value()
            config['max_depth'] = d if d > 0 else None
            config['min_samples_split'] = self.spin_min_split_rf.value()
            config['min_samples_leaf'] = self.spin_min_leaf_rf.value()
            config['bootstrap'] = self.chk_bootstrap_rf.isChecked()
            
        elif algo == "Gradient Boosting":
            config['n_estimators'] = self.spin_est_gb.value()
            config['learning_rate'] = self.spin_lr_gb.value()
            config['max_depth'] = self.spin_depth_gb.value()
            config['subsample'] = self.spin_subsample_gb.value()
            config['loss'] = self.combo_loss_gb.currentText()
            
        elif algo == "Gaussian Process":
            config['alpha'] = self.spin_alpha_gp.value()
            config['n_restarts_optimizer'] = self.spin_restarts_gp.value()
            config['normalize_y'] = self.chk_normalize_gp.isChecked()
            
        elif algo == "Deep Neural Network (PyTorch)":
            config['epochs'] = self.spin_epochs.value()
            config['learning_rate'] = self.spin_lr_pytorch.value()
            config['batch_size'] = self.spin_batch_size.value()
            config['hidden_layers'] = self.txt_hidden_layers.text()
            config['optimizer'] = self.combo_optimizer.currentText()
            config['activation'] = self.combo_pt_activation.currentText()
            config['dropout'] = self.spin_pt_dropout.value()
            config['n_mc_samples'] = self.spin_mc_samples.value()
        
        # Add debug mode setting
        config['debug_mode'] = self.radio_debug.isChecked()
            
        return config

    def start_training(self) -> None:
        if not hasattr(self, 'X_train') or self.X_train is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No training data available. Please generate or upload data first.")
            return

        config = self.get_config()
        
        self.btn_train.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.plot_widget.clear()
        self.curve_plot.clear()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.tab_widget.setCurrentWidget(self.curve_tab)  # Switch to Learning Curves tab
        
        # Recreate curves after clearing
        self.train_curve = self.curve_plot.plot(pen=pg.mkPen('r', width=2), name='Train Loss')
        self.val_curve = self.curve_plot.plot(pen=pg.mkPen('g', width=2), name='Val Loss')
        
        # Show training message for models without real-time loss curves
        model_type = config.get('model_type', 'MLP Regressor')
        if model_type != 'Deep Neural Network (PyTorch)':
            self.progress_text.setText(f"Training {model_type}...\n(Learning curves available after completion)")
            self.progress_text.show()
            # Set plot range to show the text
            self.curve_plot.setXRange(-1, 1)
            self.curve_plot.setYRange(-1, 1)
        else:
            self.progress_text.hide()
        
        # --- START WORKER (BACKGROUND THREAD) ---
        self.worker = ModelTrainingWorker(
            self.X_train, self.y_train, self.X_test, self.y_test, config
        )
        self.worker.progress_sig.connect(self.update_progress)
        self.worker.loss_sig.connect(self.update_loss_plot)
        self.worker.done_sig.connect(self.training_finished)
        self.worker.start()
        self.btn_stop.setEnabled(True)

    def update_progress(self, val, msg):
        self.progress.setValue(val)
        self.lbl_metrics.setText(msg)

    def training_finished(self, model, metrics, error):
        self.btn_train.setEnabled(True)
        self.btn_overfit1.setEnabled(True)
        self.btn_overfit10.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("Stop Training")
        
        # Hide progress text
        self.progress_text.hide()
        
        if error:
            QtWidgets.QMessageBox.critical(self, "Training Failed", error)
            self.lbl_metrics.setText("Error occurred.")
            return
            
        self.current_model = model
        self.current_metrics = metrics
        self.btn_save.setEnabled(True)
        self.progress.setValue(100)
        
        # --- FIX 1: Convert lists back to numpy arrays for plotting ---
        y_test = np.array(metrics['y_test'])
        y_pred = np.array(metrics['y_pred'])
        y_std = np.array(metrics['y_std']) if 'y_std' in metrics and metrics['y_std'] is not None else None
        
        # --- Learning Curve Display ---
        # Loss curves are already collected during training via callbacks
        # No need to dig into internal model attributes
        
        # Update the plot if we have loss data
        if self.epochs and self.train_losses:
            # Ensure curves exist
            if not hasattr(self, 'train_curve') or self.train_curve not in self.curve_plot.items():
                self.train_curve = self.curve_plot.plot(pen=pg.mkPen('r', width=2), name='Train Loss')
            if not hasattr(self, 'val_curve') or self.val_curve not in self.curve_plot.items():
                self.val_curve = self.curve_plot.plot(pen=pg.mkPen('g', width=2), name='Val Loss')
            
            self.train_curve.setData(np.array(self.epochs), np.array(self.train_losses))
            if self.val_losses and len(self.val_losses) == len(self.train_losses):
                self.val_curve.setData(np.array(self.epochs), np.array(self.val_losses))
            
            self.curve_plot.setTitle("Learning Curve")
            self.curve_plot.update()
            self.curve_plot.autoRange()

        # Update metrics text
        msg = f"<b>Training Complete</b><br>RMSE: {metrics['RMSE']:.4f}<br>R² Score: {metrics['R2']:.4f}"
        if y_std is not None:
            msg += f"<br>Mean Uncertainty: {np.mean(y_std):.4f}"
        
        # Add warning if debug mode was used
        if metrics.get('debug_mode', False):
            msg += "<br><span style='color: red; font-weight: bold;'>⚠️ DEBUG MODE - PERFECT SCORES EXPECTED<br>DO NOT USE FOR REAL DESIGN!</span>"
        
        self.lbl_metrics.setText(msg)
        
        # Handle multi-output visualization
        self.plot_widget.clear()
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            # Multi-output: plot each output separately with different colors
            colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)]
            for i in range(min(y_test.shape[1], len(colors))):
                y_test_i = y_test[:, i]
                y_pred_i = y_pred[:, i]
                color = colors[i % len(colors)]
                
                # Plot points for this output
                self.plot_widget.plot(y_test_i, y_pred_i, pen=None, 
                                    symbol='o', symbolSize=5, 
                                    symbolBrush=color + (150,),
                                    name=f'Output {i+1}')
                
                # Add diagonal line for this output
                if len(y_test_i) > 0:
                    mn, mx = float(np.min(y_test_i)), float(np.max(y_test_i))
                    self.plot_widget.plot([mn, mx], [mn, mx], 
                                        pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine))
            
            self.plot_widget.setTitle(f"Parity Plot (Multi-Output) - R² = {metrics['R2']:.4f}")
            self.plot_widget.addLegend()
        else:
            # Single output: flatten if needed and plot normally
            if y_test.ndim > 1:
                y_test = y_test.flatten()
                y_pred = y_pred.flatten()
                if y_std is not None and y_std.ndim > 1:
                    y_std = y_std.flatten()
            
            self.plot_widget.plot(y_test, y_pred, pen=None, symbol='o', symbolSize=5, symbolBrush=(100, 100, 255, 150))
            
            # Add diagonal line
            if len(y_test) > 0:
                mn, mx = float(np.min(y_test)), float(np.max(y_test))
                self.plot_widget.plot([mn, mx], [mn, mx], pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
            
            self.plot_widget.setTitle(f"Parity Plot (Predicted vs Actual) - R² = {metrics['R2']:.4f}")
        
        # Add uncertainty bands (mostly for Gaussian Process) - only for single output
        if y_std is not None and len(y_std) > 0 and (y_test.ndim == 1 or (y_test.ndim == 2 and y_test.shape[1] == 1)):
            if y_test.ndim > 1:
                y_test_flat = y_test.flatten()
                y_pred_flat = y_pred.flatten()
                y_std_flat = y_std.flatten()
            else:
                y_test_flat, y_pred_flat, y_std_flat = y_test, y_pred, y_std
                
            sort_idx = np.argsort(y_test_flat)
            y_test_sorted = y_test_flat[sort_idx]
            y_pred_sorted = y_pred_flat[sort_idx]
            y_std_sorted = y_std_flat[sort_idx]
            
            upper = y_pred_sorted + 2 * y_std_sorted
            lower = y_pred_sorted - 2 * y_std_sorted
            
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(y_test_sorted, upper, pen='g'),
                pg.PlotDataItem(y_test_sorted, lower, pen='g'),
                brush=(0, 255, 0, 50)
            )
            self.plot_widget.addItem(fill)
            self.plot_widget.plot(y_test_sorted, y_pred_sorted, pen=pg.mkPen('g', width=2))

    def save_model(self) -> None:
        if not self.current_model: return
        
        idx = self.combo_nodes.currentIndex()
        target_node = self.combo_nodes.itemData(idx)
        
        # Generate filename based on node ID, save in user data directory
        safe_id = target_node.id.replace('-', '_')
        
        # FIX: Save to local project directory 'data_surrogate' instead of AppData
        # This ensures portability and visibility
        # Try to find project root by going up from this file
        # pylcss/surrogate_modeling/surrogate_interface.py -> pylcss/surrogate_modeling -> pylcss -> root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        folder = os.path.join(base_dir, 'data_surrogate')
        
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError:
                # Fallback to current working directory if permission denied
                folder = os.path.join(os.getcwd(), 'data_surrogate')
                os.makedirs(folder, exist_ok=True)

        fname = os.path.join(folder, f"surrogate_{safe_id}.joblib")
        
        try:
            joblib.dump(self.current_model, fname)
            
            # Update Node Properties automatically
            target_node.set_property('surrogate_model_path', fname)
            target_node.set_property('use_surrogate', True)
            target_node.set_property('surrogate_status', 
                f"Trained ({self.combo_algo.currentText()}, RÂ²={self.current_metrics['R2']:.2f})")
            
            QtWidgets.QMessageBox.information(self, "Success", 
                f"Model saved to '{fname}' and attached to node.\n"
                "The node is now set to use the surrogate model.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def save_to_folder(self, folder_path):
        """Save surrogate training settings to a folder."""
        import json
        import os
        
        json_path = os.path.join(folder_path, 'surrogate_settings.json')
        
        data = {
            'target_node_index': self.combo_nodes.currentIndex(),
            'data_source': 'generate' if self.radio_gen.isChecked() else 'upload',
            'n_samples': self.spin_samples.value(),
            'algorithm_index': self.combo_algo.currentIndex(),
            # MLP
            'mlp_layers': self.txt_layers.text(),
            'mlp_activation': self.combo_activ.currentIndex(),
            'mlp_solver': self.combo_solver.currentIndex(),
            'mlp_alpha': self.spin_alpha_mlp.value(),
            'mlp_max_iter': self.spin_max_iter.value(),
            'mlp_early_stopping': self.chk_early_stopping.isChecked(),
            # RF
            'rf_estimators': self.spin_est_rf.value(),
            'rf_depth': self.spin_depth_rf.value(),
            'rf_min_split': self.spin_min_split_rf.value(),
            'rf_min_leaf': self.spin_min_leaf_rf.value(),
            'rf_bootstrap': self.chk_bootstrap_rf.isChecked(),
            # GB
            'gb_estimators': self.spin_est_gb.value(),
            'gb_lr': self.spin_lr_gb.value(),
            'gb_depth': self.spin_depth_gb.value(),
            'gb_subsample': self.spin_subsample_gb.value(),
            'gb_loss': self.combo_loss_gb.currentIndex(),
            # GP
            'gp_alpha': self.spin_alpha_gp.value(),
            'gp_restarts': self.spin_restarts_gp.value(),
            'gp_normalize': self.chk_normalize_gp.isChecked(),
            # PyTorch
            'pytorch_lr': self.spin_lr_pytorch.value(),
            'pytorch_batch': self.spin_batch_size.value(),
            'pytorch_layers': self.txt_hidden_layers.text()
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_folder(self, folder_path):
        """Load surrogate training settings from a folder."""
        import json
        import os
        
        json_path = os.path.join(folder_path, 'surrogate_settings.json')
        if not os.path.exists(json_path):
            return
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            self.combo_nodes.setCurrentIndex(data.get('target_node_index', 0))
            if data.get('data_source') == 'generate':
                self.radio_gen.setChecked(True)
            else:
                self.radio_upload.setChecked(True)
                
            self.spin_samples.setValue(data.get('n_samples', 1000))
            self.combo_algo.setCurrentIndex(data.get('algorithm_index', 0))
            
            # MLP
            self.txt_layers.setText(data.get('mlp_layers', '(100, 50)'))
            self.combo_activ.setCurrentIndex(data.get('mlp_activation', 0))
            self.combo_solver.setCurrentIndex(data.get('mlp_solver', 0))
            self.spin_alpha_mlp.setValue(data.get('mlp_alpha', 0.0001))
            self.spin_max_iter.setValue(data.get('mlp_max_iter', 5000))
            self.chk_early_stopping.setChecked(data.get('mlp_early_stopping', False))
            
            # RF
            self.spin_est_rf.setValue(data.get('rf_estimators', 100))
            self.spin_depth_rf.setValue(data.get('rf_depth', 0))
            self.spin_min_split_rf.setValue(data.get('rf_min_split', 2))
            self.spin_min_leaf_rf.setValue(data.get('rf_min_leaf', 1))
            self.chk_bootstrap_rf.setChecked(data.get('rf_bootstrap', True))
            
            # GB
            self.spin_est_gb.setValue(data.get('gb_estimators', 100))
            self.spin_lr_gb.setValue(data.get('gb_lr', 0.1))
            self.spin_depth_gb.setValue(data.get('gb_depth', 3))
            self.spin_subsample_gb.setValue(data.get('gb_subsample', 1.0))
            self.combo_loss_gb.setCurrentIndex(data.get('gb_loss', 0))
            
            # GP
            self.spin_alpha_gp.setValue(data.get('gp_alpha', 1e-6))
            self.spin_restarts_gp.setValue(data.get('gp_restarts', 15))
            self.chk_normalize_gp.setChecked(data.get('gp_normalize', True))
            
            # PyTorch
            self.spin_lr_pytorch.setValue(data.get('pytorch_lr', 0.01))
            self.spin_batch_size.setValue(data.get('pytorch_batch', 32))
            self.txt_hidden_layers.setText(data.get('pytorch_layers', '64, 64'))
            
        except Exception as e:
            logger.exception("Failed to load surrogate settings")

# Worker Thread class to bridge GUI and Engine
class DataGenerationWorker(QtCore.QThread):
    progress_sig = QtCore.Signal(int, str)
    done_sig = QtCore.Signal(object, str)
    
    def __init__(self, spy_code: str, spy_inputs: List[str], spy_outputs: List[str], input_bounds: List[Tuple[float, float]], samples: int) -> None:
        super().__init__()
        self.spy_code = spy_code
        self.spy_inputs = spy_inputs
        self.spy_outputs = spy_outputs
        self.input_bounds = input_bounds
        self.samples = samples
        self.trainer = SurrogateTrainer()
        
    def run(self) -> None:
        try:
            # Generate
            X_train, y_train, X_test, y_test, _, _ = self.trainer.generate_data(
                self.spy_code, self.spy_inputs, self.spy_outputs, self.input_bounds,
                self.samples, 
                callback=lambda p, m: self.progress_sig.emit(p, m)
            )
            self.done_sig.emit((X_train, y_train, X_test, y_test), None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.done_sig.emit(None, str(e))

class ModelTrainingWorker(QtCore.QThread):
    progress_sig = QtCore.Signal(int, str)
    loss_sig = QtCore.Signal(dict)
    done_sig = QtCore.Signal(object, object, str)
    
    def __init__(self, X_train, y_train, X_test, y_test, config: Dict[str, Any]) -> None:
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.trainer = SurrogateTrainer()
        self.stop_flag = False
        
    def run(self) -> None:
        try:
            # Train
            model, metrics = self.trainer.train_model(
                self.X_train, self.y_train, self.config, self.X_test, self.y_test,
                callback=lambda p, m: self.progress_sig.emit(p, m),
                stop_flag=lambda: self.stop_flag,
                loss_callback=lambda d: self.loss_sig.emit(d)
            )
            
            self.done_sig.emit(model, metrics, None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.done_sig.emit(None, None, str(e))

class TrainingWorker(QtCore.QThread):
    progress_sig = QtCore.Signal(int, str)
    loss_sig = QtCore.Signal(dict)
    done_sig = QtCore.Signal(object, object, str)
    
    def __init__(self, spy_code, spy_inputs, spy_outputs, input_bounds, samples, config):
        super().__init__()
        self.spy_code = spy_code
        self.spy_inputs = spy_inputs
        self.spy_outputs = spy_outputs
        self.input_bounds = input_bounds
        self.samples = samples
        self.config = config
        self.trainer = SurrogateTrainer()
        self.stop_flag = False

    def run(self):
        try:
            # 1. Generate Data
            self.progress_sig.emit(0, "Generating debug data...")
            X_train, y_train, X_test, y_test, _, _ = self.trainer.generate_data(
                self.spy_code, self.spy_inputs, self.spy_outputs, self.input_bounds,
                self.samples, 
                callback=None
            )
            
            # 2. Train Model
            self.progress_sig.emit(20, "Training debug model...")
            model, metrics = self.trainer.train_model(
                X_train, y_train, self.config, X_test, y_test,
                callback=lambda p, m: self.progress_sig.emit(20 + int(p*0.8), m),
                stop_flag=lambda: self.stop_flag,
                loss_callback=lambda d: self.loss_sig.emit(d)
            )
            
            self.done_sig.emit(model, metrics, None)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.done_sig.emit(None, None, str(e))
