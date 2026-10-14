# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Interactive Custom Architecture Builder for deep surrogate modeling.
Provides both a Visual Layer Stack Builder and a raw PyTorch Code Editor."""

from __future__ import annotations

import logging
import math
from typing import Any

from PySide6 import QtCore, QtWidgets

from pylcss.mdao.surrogate_modeling.custom_architecture import (
    compile_custom_code,
)
from pylcss.mdao.surrogate_modeling.torch_utils import TORCH_AVAILABLE

logger = logging.getLogger(__name__)

__all__ = ["ArchitectureBuilderWidget"]

DEFAULT_CODE_TEMPLATE = '''import torch
import torch.nn as nn

class CustomSurrogate(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.net(x)
'''


class ArchitectureBuilderWidget(QtWidgets.QWidget):
    """Widget allowing users to visually build or script custom deep surrogate architectures."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.custom_layers: list[dict[str, Any]] = []
        self._setup_ui()
        self._apply_preset_by_name("Deep ResNet (Recommended)")

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.sub_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.sub_tabs)

        # Tab 1: Visual Layer Builder
        tab_visual = QtWidgets.QWidget()
        l_vis = QtWidgets.QVBoxLayout(tab_visual)

        # Preset selection
        row_preset = QtWidgets.QHBoxLayout()
        lbl_pre = QtWidgets.QLabel("Architecture Preset:")
        self.combo_presets = QtWidgets.QComboBox()
        self.combo_presets.addItem("Deep ResNet (Recommended)")
        self.combo_presets.addItem("Fourier Physics Network (Multiscale)")
        self.combo_presets.addItem("Dense Highway MLP (Wide and Deep)")
        self.combo_presets.addItem("Standard Multi-Layer Perceptron")
        self.combo_presets.addItem("Custom / Blank")
        self.combo_presets.currentTextChanged.connect(self._apply_preset_by_name)
        row_preset.addWidget(lbl_pre)
        row_preset.addWidget(self.combo_presets, 1)
        l_vis.addLayout(row_preset)

        # Fourier Features input embedding
        grp_fourier = QtWidgets.QGroupBox("Fourier Feature Embeddings")
        l_four = QtWidgets.QVBoxLayout(grp_fourier)
        l_four.setContentsMargins(6, 6, 6, 6)
        l_four.setSpacing(4)
        self.chk_fourier = QtWidgets.QCheckBox("Enable Multiscale Fourier Features")
        self.chk_fourier.toggled.connect(self._update_summary)
        l_four.addWidget(self.chk_fourier)

        row_fparams = QtWidgets.QHBoxLayout()
        row_fparams.setSpacing(6)
        lbl_fn = QtWidgets.QLabel("Features:")
        self.spin_fourier_num = QtWidgets.QSpinBox()
        self.spin_fourier_num.setRange(8, 256)
        self.spin_fourier_num.setValue(32)
        lbl_fs = QtWidgets.QLabel("Scale:")
        self.spin_fourier_scale = QtWidgets.QDoubleSpinBox()
        self.spin_fourier_scale.setRange(0.01, 50.0)
        self.spin_fourier_scale.setValue(1.0)
        row_fparams.addWidget(lbl_fn)
        row_fparams.addWidget(self.spin_fourier_num, 1)
        row_fparams.addWidget(lbl_fs)
        row_fparams.addWidget(self.spin_fourier_scale, 1)
        l_four.addLayout(row_fparams)
        l_vis.addWidget(grp_fourier)

        # Layer stack table
        self.table_layers = QtWidgets.QTableWidget()
        self.table_layers.setColumnCount(4)
        self.table_layers.setHorizontalHeaderLabels(["Layer Type", "Units", "Activation", "Dropout"])
        self.table_layers.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table_layers.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.table_layers.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table_layers.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.table_layers.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_layers.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table_layers.setMinimumHeight(130)
        self.table_layers.itemChanged.connect(self._on_layer_edited)
        l_vis.addWidget(self.table_layers)

        # Layer manipulation buttons in clean 2-row grid
        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setContentsMargins(0, 2, 0, 2)
        btn_grid.setSpacing(4)
        self.btn_add_linear = QtWidgets.QPushButton("+ Linear")
        self.btn_add_res = QtWidgets.QPushButton("+ ResBlock")
        self.btn_add_norm = QtWidgets.QPushButton("+ LayerNorm")
        self.btn_add_drop = QtWidgets.QPushButton("+ Dropout")
        self.btn_remove = QtWidgets.QPushButton("Remove")

        self.btn_add_linear.clicked.connect(lambda: self._add_layer("linear", 128, "gelu", 0.0))
        self.btn_add_res.clicked.connect(lambda: self._add_layer("resblock", 128, "gelu", 0.05))
        self.btn_add_norm.clicked.connect(lambda: self._add_layer("layernorm", 128, "none", 0.0))
        self.btn_add_drop.clicked.connect(lambda: self._add_layer("dropout", 0, "none", 0.1))
        self.btn_remove.clicked.connect(self._remove_selected_layer)

        btn_grid.addWidget(self.btn_add_linear, 0, 0)
        btn_grid.addWidget(self.btn_add_res, 0, 1)
        btn_grid.addWidget(self.btn_add_norm, 0, 2)
        btn_grid.addWidget(self.btn_add_drop, 1, 0)
        btn_grid.addWidget(self.btn_remove, 1, 1, 1, 2)
        l_vis.addLayout(btn_grid)

        # Summary box
        self.lbl_vis_summary = QtWidgets.QLabel("Architecture: 0 layers.")
        self.lbl_vis_summary.setStyleSheet("color: #b0b0b0; font-size: 8.5pt;")
        l_vis.addWidget(self.lbl_vis_summary)

        self.sub_tabs.addTab(tab_visual, "Visual Layer Builder")

        # Tab 2: PyTorch Code Editor
        tab_code = QtWidgets.QWidget()
        l_code = QtWidgets.QVBoxLayout(tab_code)

        lbl_code_hint = QtWidgets.QLabel(
            "Define your custom torch.nn.Module subclass. "
            "Constructor receives (in_features, out_features)."
        )
        lbl_code_hint.setWordWrap(True)
        lbl_code_hint.setStyleSheet("color: #b0b0b0; font-size: 8.5pt;")
        l_code.addWidget(lbl_code_hint)

        self.txt_code = QtWidgets.QPlainTextEdit()
        self.txt_code.setPlainText(DEFAULT_CODE_TEMPLATE)
        self.txt_code.setMinimumHeight(160)
        l_code.addWidget(self.txt_code)

        row_val = QtWidgets.QHBoxLayout()
        self.btn_validate_code = QtWidgets.QPushButton("Validate Code")
        self.btn_validate_code.clicked.connect(self.validate_custom_code)
        self.lbl_validation_status = QtWidgets.QLabel("Status: Not validated.")
        row_val.addWidget(self.btn_validate_code)
        row_val.addWidget(self.lbl_validation_status, 1)
        l_code.addLayout(row_val)

        self.sub_tabs.addTab(tab_code, "PyTorch Code Editor")

        # --- Training Hyperparameters Form (2-column compact grid) ---
        grp_train = QtWidgets.QGroupBox("Training Hyperparameters")
        g_train = QtWidgets.QGridLayout(grp_train)
        g_train.setContentsMargins(6, 6, 6, 6)
        g_train.setSpacing(4)

        self.combo_optimizer = QtWidgets.QComboBox()
        self.combo_optimizer.addItems(["AdamW", "Adam", "SGD", "RMSprop"])
        g_train.addWidget(QtWidgets.QLabel("Optimizer:"), 0, 0)
        g_train.addWidget(self.combo_optimizer, 0, 1)

        self.spin_lr = QtWidgets.QDoubleSpinBox()
        self.spin_lr.setRange(1e-6, 1.0)
        self.spin_lr.setDecimals(6)
        self.spin_lr.setValue(0.001)
        self.spin_lr.setSingleStep(0.0005)
        g_train.addWidget(QtWidgets.QLabel("Learning Rate:"), 0, 2)
        g_train.addWidget(self.spin_lr, 0, 3)

        self.spin_decay = QtWidgets.QDoubleSpinBox()
        self.spin_decay.setRange(0.0, 0.1)
        self.spin_decay.setDecimals(5)
        self.spin_decay.setValue(0.0001)
        self.spin_decay.setSingleStep(0.0001)
        g_train.addWidget(QtWidgets.QLabel("Weight Decay:"), 1, 0)
        g_train.addWidget(self.spin_decay, 1, 1)

        self.spin_batch_size = QtWidgets.QSpinBox()
        self.spin_batch_size.setRange(8, 2048)
        self.spin_batch_size.setValue(32)
        self.spin_batch_size.setSingleStep(16)
        g_train.addWidget(QtWidgets.QLabel("Batch Size:"), 1, 2)
        g_train.addWidget(self.spin_batch_size, 1, 3)

        self.spin_epochs = QtWidgets.QSpinBox()
        self.spin_epochs.setRange(10, 50000)
        self.spin_epochs.setValue(1000)
        self.spin_epochs.setSingleStep(200)
        g_train.addWidget(QtWidgets.QLabel("Epochs:"), 2, 0)
        g_train.addWidget(self.spin_epochs, 2, 1)

        self.spin_patience = QtWidgets.QSpinBox()
        self.spin_patience.setRange(5, 500)
        self.spin_patience.setValue(40)
        g_train.addWidget(QtWidgets.QLabel("Patience:"), 2, 2)
        g_train.addWidget(self.spin_patience, 2, 3)

        self.spin_mc_samples = QtWidgets.QSpinBox()
        self.spin_mc_samples.setRange(10, 500)
        self.spin_mc_samples.setValue(50)
        g_train.addWidget(QtWidgets.QLabel("MC Samples:"), 3, 0)
        g_train.addWidget(self.spin_mc_samples, 3, 1)

        layout.addWidget(grp_train)

    def _apply_preset_by_name(self, name: str) -> None:
        self.custom_layers.clear()
        if "ResNet" in name:
            self.chk_fourier.setChecked(False)
            self.custom_layers = [
                {"type": "linear", "units": 128, "activation": "gelu", "dropout": 0.0},
                {"type": "resblock", "units": 128, "activation": "gelu", "dropout": 0.05},
                {"type": "resblock", "units": 128, "activation": "gelu", "dropout": 0.05},
                {"type": "linear", "units": 64, "activation": "gelu", "dropout": 0.0},
            ]
        elif "Fourier" in name:
            self.chk_fourier.setChecked(True)
            self.spin_fourier_num.setValue(32)
            self.spin_fourier_scale.setValue(1.0)
            self.custom_layers = [
                {"type": "linear", "units": 128, "activation": "silu", "dropout": 0.0},
                {"type": "linear", "units": 128, "activation": "silu", "dropout": 0.0},
                {"type": "linear", "units": 64, "activation": "silu", "dropout": 0.0},
            ]
        elif "Highway" in name:
            self.chk_fourier.setChecked(False)
            self.custom_layers = [
                {"type": "linear", "units": 256, "activation": "relu", "dropout": 0.1},
                {"type": "layernorm", "units": 256, "activation": "none", "dropout": 0.0},
                {"type": "linear", "units": 128, "activation": "relu", "dropout": 0.1},
                {"type": "layernorm", "units": 128, "activation": "none", "dropout": 0.0},
                {"type": "linear", "units": 64, "activation": "relu", "dropout": 0.05},
            ]
        elif "Standard" in name:
            self.chk_fourier.setChecked(False)
            self.custom_layers = [
                {"type": "linear", "units": 100, "activation": "relu", "dropout": 0.0},
                {"type": "linear", "units": 50, "activation": "relu", "dropout": 0.0},
            ]
        else:
            self.chk_fourier.setChecked(False)

        self._refresh_table()

    def _add_layer(self, layer_type: str, units: int, activation: str, dropout: float) -> None:
        self.custom_layers.append(
            {"type": layer_type, "units": units, "activation": activation, "dropout": dropout}
        )
        self._refresh_table()

    def _remove_selected_layer(self) -> None:
        row = self.table_layers.currentRow()
        if 0 <= row < len(self.custom_layers):
            del self.custom_layers[row]
            self._refresh_table()

    def _refresh_table(self) -> None:
        blocker = QtCore.QSignalBlocker(self.table_layers)
        self.table_layers.setRowCount(len(self.custom_layers))
        for r, layer in enumerate(self.custom_layers):
            l_type = str(layer.get("type", "")).upper()
            units = str(layer.get("units", "-"))
            act = str(layer.get("activation", "none")).upper()
            drop = str(layer.get("rate", layer.get("dropout", 0.0)))

            self.table_layers.setItem(r, 0, QtWidgets.QTableWidgetItem(l_type))
            self.table_layers.setItem(r, 1, QtWidgets.QTableWidgetItem(units if l_type in ("LINEAR", "RESBLOCK") else "-"))
            self.table_layers.setItem(r, 2, QtWidgets.QTableWidgetItem(act if l_type in ("LINEAR", "RESBLOCK") else "-"))
            self.table_layers.setItem(r, 3, QtWidgets.QTableWidgetItem(drop if float(drop) > 0 else "-"))
            for column in range(4):
                item = self.table_layers.item(r, column)
                editable = (
                    column in (1, 2, 3) and l_type in ("LINEAR", "RESBLOCK")
                ) or (column == 3 and l_type == "DROPOUT")
                if not editable:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)

        del blocker
        self._update_summary()

    def _on_layer_edited(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Keep the architecture used for training in sync with visible edits."""
        row, column = item.row(), item.column()
        if not 0 <= row < len(self.custom_layers):
            return
        layer = self.custom_layers[row]
        layer_type = str(layer.get("type", "")).lower()
        text = item.text().strip().lower()
        try:
            if column == 1 and layer_type in {"linear", "resblock"}:
                units = int(text)
                if units < 1:
                    raise ValueError("Units must be a positive integer.")
                layer["units"] = units
            elif column == 2 and layer_type in {"linear", "resblock"}:
                activations = {
                    "relu", "gelu", "silu", "swish", "mish", "tanh",
                    "leaky_relu", "leakyrelu", "elu", "sigmoid", "logistic",
                    "identity", "none", "linear",
                }
                if text not in activations:
                    raise ValueError("Enter a supported activation, such as GELU or RELU.")
                layer["activation"] = text
            elif column == 3 and layer_type in {"linear", "resblock", "dropout"}:
                rate = 0.0 if text in {"", "-"} else float(text)
                if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
                    raise ValueError("Dropout must be in [0, 1).")
                layer["dropout"] = rate
                if layer_type == "dropout":
                    layer["rate"] = rate
            else:
                return
        except ValueError as exc:
            self._refresh_table()
            self.lbl_vis_summary.setText(str(exc))
            self.lbl_vis_summary.setToolTip(str(exc))
            return
        self.lbl_vis_summary.setToolTip("")
        self._update_summary()

    def _update_summary(self) -> None:
        count = len(self.custom_layers)
        fourier_str = "with Fourier Embeddings" if self.chk_fourier.isChecked() else "standard input"
        self.lbl_vis_summary.setText(f"Architecture: {count} layers configured ({fourier_str}).")

    def validate_custom_code(self) -> bool:
        if not TORCH_AVAILABLE:
            self.lbl_validation_status.setText("Error: PyTorch not installed.")
            self.lbl_validation_status.setStyleSheet("color: #ff8888;")
            return False

        code_str = self.txt_code.toPlainText().strip()
        try:
            compile_custom_code(code_str, in_features=4, out_features=1)
            self.lbl_validation_status.setText("Model code validated successfully.")
            self.lbl_validation_status.setStyleSheet("color: #a0ffa0;")
            return True
        except Exception as e:
            self.lbl_validation_status.setText(f"Validation failed: {e}")
            self.lbl_validation_status.setStyleSheet("color: #ff8888;")
            return False

    def get_config(self) -> dict[str, Any]:
        """Return full configuration dictionary for CustomArchitectureStrategy."""
        mode = "visual" if self.sub_tabs.currentIndex() == 0 else "code"
        return {
            "custom_arch_mode": mode,
            "custom_spec": {
                "fourier_features": {
                    "enabled": self.chk_fourier.isChecked(),
                    "num_features": self.spin_fourier_num.value(),
                    "scale": self.spin_fourier_scale.value(),
                },
                "layers": list(self.custom_layers),
            },
            "custom_code": self.txt_code.toPlainText().strip(),
            "optimizer": self.combo_optimizer.currentText(),
            "learning_rate": self.spin_lr.value(),
            "weight_decay": self.spin_decay.value(),
            "epochs": self.spin_epochs.value(),
            "batch_size": self.spin_batch_size.value(),
            "patience": self.spin_patience.value(),
            "n_mc_samples": self.spin_mc_samples.value(),
        }

    def set_config(self, cfg: dict[str, Any]) -> None:
        """Load configuration dictionary into UI."""
        if not cfg:
            return
        mode = cfg.get("custom_arch_mode", "visual")
        self.sub_tabs.setCurrentIndex(0 if mode == "visual" else 1)

        spec = cfg.get("custom_spec", {})
        fourier = spec.get("fourier_features", {})
        self.chk_fourier.setChecked(fourier.get("enabled", False))
        self.spin_fourier_num.setValue(fourier.get("num_features", 32))
        self.spin_fourier_scale.setValue(fourier.get("scale", 1.0))

        if "layers" in spec:
            self.custom_layers = list(spec["layers"])
            self._refresh_table()

        if "custom_code" in cfg:
            self.txt_code.setPlainText(cfg["custom_code"])

        if "optimizer" in cfg:
            idx = self.combo_optimizer.findText(cfg["optimizer"])
            if idx >= 0:
                self.combo_optimizer.setCurrentIndex(idx)
        if "learning_rate" in cfg:
            self.spin_lr.setValue(float(cfg["learning_rate"]))
        if "weight_decay" in cfg:
            self.spin_decay.setValue(float(cfg["weight_decay"]))
        if "epochs" in cfg:
            self.spin_epochs.setValue(int(cfg["epochs"]))
        if "batch_size" in cfg:
            self.spin_batch_size.setValue(int(cfg["batch_size"]))
        if "patience" in cfg:
            self.spin_patience.setValue(int(cfg["patience"]))
        if "n_mc_samples" in cfg:
            self.spin_mc_samples.setValue(int(cfg["n_mc_samples"]))
