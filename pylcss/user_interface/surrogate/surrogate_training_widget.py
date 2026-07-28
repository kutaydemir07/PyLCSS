# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Surrogate-training widget assembled from focused workflow mixins."""

from __future__ import annotations

from typing import Any

from PySide6 import QtWidgets

from pylcss.surrogate_modeling.training_engine import SKLEARN_AVAILABLE
from pylcss.user_interface.common.theme_manager import COLORS

from .data_workflow import SurrogateDataMixin
from .evaluation import SurrogateEvaluationMixin
from .persistence import SurrogatePersistenceMixin
from .training_workflow import SurrogateTrainingMixin
from .ui_setup import SurrogateUiMixin

__all__ = ["SurrogateTrainingWidget"]


class SurrogateTrainingWidget(
    SurrogateUiMixin,
    SurrogateDataMixin,
    SurrogateTrainingMixin,
    SurrogateEvaluationMixin,
    SurrogatePersistenceMixin,
    QtWidgets.QWidget,
):
    """Main workflow widget for surrogate-model training."""

    _FIELD_CHOICES_BY_SOLVER = {
        "fea": ["von_mises", "displacement", "stress_tensor", "ener_nodal"],
        "crash": ["stress_vm", "displacement", "von_mises"],
        "topopt": ["density"],
    }

    def apply_theme(self, _theme_name):
        """Keep engineering plots paper-white with dark, printable labels."""
        foreground = COLORS["chart_fg"]
        for plot in (self.curve_plot, self.plot_widget, self.fi_plot):
            plot.setBackground(COLORS["chart_bg"])
            item = plot.getPlotItem()
            for axis_name in ("left", "right", "top", "bottom"):
                axis = item.getAxis(axis_name)
                axis.setPen(foreground)
                axis.setTextPen(foreground)
            item.titleLabel.setAttr("color", foreground)
        self.progress_text.setColor(foreground)

    def __init__(self, modeling_widget: QtWidgets.QWidget | None = None) -> None:
        super().__init__()
        self.modeling_widget = modeling_widget
        self.current_model: Any | None = None
        self.current_metrics: dict[str, Any] | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.epochs: list[int] = []

        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.setup_ui()

        if not SKLEARN_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Dependency",
                "Scikit-learn is required for this feature.\nPlease install it: pip install scikit-learn",
            )
            self.setEnabled(False)
