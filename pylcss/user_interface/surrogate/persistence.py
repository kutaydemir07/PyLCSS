# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SurrogatePersistenceMixin behavior for surrogate training."""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)

__all__ = ["SurrogatePersistenceMixin"]


class SurrogatePersistenceMixin:
    def save_to_folder(self, folder_path):
        """Save surrogate training settings to a folder."""
        from pylcss.io_manager.project_io import atomic_json_dump

        json_path = os.path.join(folder_path, "surrogate_settings.json")

        data = {
            "target_node_index": self.combo_nodes.currentIndex(),
            "data_source": "generate" if self.radio_gen.isChecked() else "upload",
            "n_samples": self.spin_samples.value(),
            "algorithm_index": self.combo_algo.currentIndex(),
            # MLP
            "mlp_layers": self.txt_layers.text(),
            "mlp_activation": self.combo_activ.currentIndex(),
            "mlp_solver": self.combo_solver.currentIndex(),
            "mlp_alpha": self.spin_alpha_mlp.value(),
            "mlp_max_iter": self.spin_max_iter.value(),
            "mlp_early_stopping": self.chk_early_stopping.isChecked(),
            # RF
            "rf_estimators": self.spin_est_rf.value(),
            "rf_depth": self.spin_depth_rf.value(),
            "rf_min_split": self.spin_min_split_rf.value(),
            "rf_min_leaf": self.spin_min_leaf_rf.value(),
            "rf_bootstrap": self.chk_bootstrap_rf.isChecked(),
            # GB
            "gb_estimators": self.spin_est_gb.value(),
            "gb_lr": self.spin_lr_gb.value(),
            "gb_depth": self.spin_depth_gb.value(),
            "gb_subsample": self.spin_subsample_gb.value(),
            "gb_loss": self.combo_loss_gb.currentIndex(),
            # GP
            "gp_alpha": self.spin_alpha_gp.value(),
            "gp_restarts": self.spin_restarts_gp.value(),
            "gp_normalize": self.chk_normalize_gp.isChecked(),
            "active_learning": self.active_learning_values(),
            # PyTorch
            "pytorch_lr": self.spin_lr_pytorch.value(),
            "pytorch_batch": self.spin_batch_size.value(),
            "pytorch_layers": self.txt_hidden_layers.text(),
        }

        atomic_json_dump(data, json_path)

    def load_from_folder(self, folder_path):
        """Load surrogate training settings from a folder."""
        from pylcss.io_manager.project_io import load_json_object

        json_path = os.path.join(folder_path, "surrogate_settings.json")
        if not os.path.exists(json_path):
            return

        try:
            data = load_json_object(json_path)

            self.combo_nodes.setCurrentIndex(data.get("target_node_index", 0))
            if data.get("data_source") == "generate":
                self.radio_gen.setChecked(True)
            else:
                self.radio_upload.setChecked(True)

            self.spin_samples.setValue(data.get("n_samples", 1000))
            self.combo_algo.setCurrentIndex(data.get("algorithm_index", 0))

            # MLP
            self.txt_layers.setText(data.get("mlp_layers", "(100, 50)"))
            self.combo_activ.setCurrentIndex(data.get("mlp_activation", 0))
            self.combo_solver.setCurrentIndex(data.get("mlp_solver", 0))
            self.spin_alpha_mlp.setValue(data.get("mlp_alpha", 0.0001))
            self.spin_max_iter.setValue(data.get("mlp_max_iter", 5000))
            self.chk_early_stopping.setChecked(data.get("mlp_early_stopping", False))

            # RF
            self.spin_est_rf.setValue(data.get("rf_estimators", 100))
            self.spin_depth_rf.setValue(data.get("rf_depth", 0))
            self.spin_min_split_rf.setValue(data.get("rf_min_split", 2))
            self.spin_min_leaf_rf.setValue(data.get("rf_min_leaf", 1))
            self.chk_bootstrap_rf.setChecked(data.get("rf_bootstrap", True))

            # GB
            self.spin_est_gb.setValue(data.get("gb_estimators", 100))
            self.spin_lr_gb.setValue(data.get("gb_lr", 0.1))
            self.spin_depth_gb.setValue(data.get("gb_depth", 3))
            self.spin_subsample_gb.setValue(data.get("gb_subsample", 1.0))
            self.combo_loss_gb.setCurrentIndex(data.get("gb_loss", 0))

            # GP
            self.spin_alpha_gp.setValue(data.get("gp_alpha", 1e-6))
            self.spin_restarts_gp.setValue(data.get("gp_restarts", 15))
            self.chk_normalize_gp.setChecked(data.get("gp_normalize", True))
            self.apply_active_learning_settings(data.get("active_learning", {}))

            # PyTorch
            self.spin_lr_pytorch.setValue(data.get("pytorch_lr", 0.01))
            self.spin_batch_size.setValue(data.get("pytorch_batch", 32))
            self.txt_hidden_layers.setText(data.get("pytorch_layers", "64, 64"))

        except Exception:
            logger.exception("Failed to load surrogate settings")
