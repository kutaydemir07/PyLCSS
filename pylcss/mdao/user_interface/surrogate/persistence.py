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
            "active_learning": self.active_learning_values(),
            # Auto Mode
            "auto_preset": (
                self.combo_auto_preset.currentData()
                if hasattr(self, "combo_auto_preset")
                else "quick"
            ),
            "auto_metric": (
                self.combo_auto_metric.currentData()
                if hasattr(self, "combo_auto_metric")
                else "R2"
            ),
            "auto_ensemble": (
                self.chk_auto_ensemble.isChecked()
                if hasattr(self, "chk_auto_ensemble")
                else True
            ),
            # Custom Architecture
            "custom_architecture": (
                self.arch_builder.get_config()
                if hasattr(self, "arch_builder")
                else {}
            ),
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

            self.apply_active_learning_settings(data.get("active_learning", {}))

            # Auto Mode
            if hasattr(self, "combo_auto_preset") and "auto_preset" in data:
                idx = self.combo_auto_preset.findData(data["auto_preset"])
                if idx >= 0:
                    self.combo_auto_preset.setCurrentIndex(idx)
            if hasattr(self, "combo_auto_metric") and "auto_metric" in data:
                idx = self.combo_auto_metric.findData(data["auto_metric"])
                if idx >= 0:
                    self.combo_auto_metric.setCurrentIndex(idx)
            if hasattr(self, "chk_auto_ensemble") and "auto_ensemble" in data:
                self.chk_auto_ensemble.setChecked(bool(data["auto_ensemble"]))

            # Custom Architecture
            if hasattr(self, "arch_builder") and "custom_architecture" in data:
                self.arch_builder.set_config(data["custom_architecture"])

        except Exception:
            logger.exception("Failed to load surrogate settings")
