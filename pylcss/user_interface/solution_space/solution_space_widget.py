# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Solution-space widget assembled from focused workflow mixins."""

from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtCore, QtWidgets

from ...solution_space.models import BoxSolutionSpace, MultiModalResult
from .adg import AllDimensionsGraphMixin
from .computation import SolutionComputationMixin
from .model_loading import SolutionModelMixin
from .persistence import SolutionPersistenceMixin
from .plot_management import SolutionPlotMixin
from .plotting import (
    ArrowLine,
    ColorConfigDialog,
    PlotWidget,
    ScalableText,
    VariantRequirementsDialog,
)
from .product_family import ProductFamilyMixin
from .requirements import SolutionRequirementsMixin
from .solver_workers import MultiModalResampleWorker, MultiModalSolverWorker
from .ui_setup import SolutionUiMixin

__all__ = [
    "ArrowLine",
    "ColorConfigDialog",
    "PlotWidget",
    "ScalableText",
    "SolutionSpaceWidget",
    "VariantRequirementsDialog",
]


class SolutionSpaceWidget(
    SolutionUiMixin,
    SolutionModelMixin,
    SolutionRequirementsMixin,
    SolutionComputationMixin,
    SolutionPlotMixin,
    SolutionPersistenceMixin,
    ProductFamilyMixin,
    AllDimensionsGraphMixin,
    QtWidgets.QWidget,
):
    """Main workflow widget for solution-space analysis."""

    def _to_serializable(self, obj):
        """Recursively convert numpy arrays in obj to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._to_serializable(v) for v in obj)
        else:
            return obj

    def trigger_debounced_resample(self):
        """Trigger resampling after a short delay to prevent freezing during rapid updates."""
        # Don't auto-resample from box/ROI changes until the user has sampled once
        # explicitly — otherwise merely loading/forwarding a model samples on render.
        if not self._has_sampled:
            return
        self.resample_timer.start()

    def _resample_current_view(self, silent=False):
        """Refresh samples with the strategy for the currently displayed view."""
        if self.multi_modal_boxes:
            self.resample_multimodal(silent=silent)
        else:
            self.resample_box(silent=silent)

    def _safe_get_float(self, item, default=0.0):
        """Safely convert table item text to float."""
        if item is None:
            return default
        text = item.text().strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the solution space analysis widget."""
        super().__init__(parent)

        self.problem = None
        self.system_code = None
        self.inputs = []
        self.outputs = []
        self.models = []
        self.dv_par_box = None
        self.last_samples = None
        self.plot_widgets = []
        self.updating_plots = False
        self._plot_export_busy = False
        self.resample_thread = None
        self.candidate_worker = None
        self.resampling = False
        self.pending_restart = False
        # Box-drag live resampling only kicks in AFTER the user has explicitly
        # sampled once (Resample/Compute). This stops a just-forwarded model —
        # whose ROI emits a region-change on first render — from auto-sampling.
        self._has_sampled = False
        self.qoi_colors = {}
        self.optimal_point = None

        # Multi-Modal solution-space state.  The controls live on their own
        # configuration tab so the established single-box workflow stays clean.
        self.multi_modal_result: MultiModalResult | None = None
        self.multi_modal_boxes: list[BoxSolutionSpace] = []
        self.active_box_index = -1
        self.multimodal_view_mode = "all"
        self.multi_modal_worker: MultiModalSolverWorker | None = None
        self.multimodal_resample_worker: MultiModalResampleWorker | None = None
        self._multimodal_resample_pending = False
        self._multimodal_resample_request = 0
        self.adg_layout_worker = None
        self.box_colors = [
            "#3498db",
            "#9b59b6",
            "#f39c12",
            "#1abc9c",
            "#e91e63",
            "#e6194b",
            "#4363d8",
            "#ffe119",
            "#f58231",
            "#911eb4",
            "#46f0f0",
            "#f032e6",
            "#9a6324",
            "#800000",
            "#000075",
        ]

        # Debounce timer for resampling
        self.resample_timer = QtCore.QTimer()
        self.resample_timer.setSingleShot(True)
        self.resample_timer.setInterval(300)  # 300ms delay
        self.resample_timer.timeout.connect(
            lambda: self._resample_current_view(silent=True)
        )

        # Thread safety: Mutex for dv_par_box access
        self.dv_par_box_mutex = QtCore.QRecursiveMutex()
        # Distinct colors excluding green (e.g. #3cb44b, #00aa00)
        self.default_colors = [
            "#e6194b",  # Red
            "#4363d8",  # Blue
            "#ffe119",  # Yellow
            "#f58231",  # Orange
            "#911eb4",  # Purple
            "#42d4f4",  # Cyan
            "#f032e6",  # Magenta
            "#a9a9a9",  # Grey
            "#fabebe",  # Pink
            "#000075",  # Navy
            "#9a6324",  # Brown
            "#800000",  # Maroon
            "#e6beff",  # Lavender
            "#fffac8",  # Beige
        ]
        self.input_units = {}
        self.output_units = {}

        # Connect to application quit to clean up threads
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.on_app_quit)

        self.init_ui()

    def _tracked_threads(self):
        threads = []
        for name in (
            "solver_worker",
            "multi_modal_worker",
            "multimodal_resample_worker",
            "family_worker",
            "resample_thread",
            "candidate_worker",
            "interpolation_thread",
            "adg_layout_worker",
        ):
            thread = getattr(self, name, None)
            if thread is not None:
                threads.append(thread)
        return threads

    def has_active_background_tasks(self):
        return any(thread.isRunning() for thread in self._tracked_threads())

    def request_background_stop(self):
        for thread in self._tracked_threads():
            try:
                if hasattr(thread, "stop"):
                    thread.stop()
                elif hasattr(thread, "cancel"):
                    thread.cancel()
                elif hasattr(thread, "requestInterruption"):
                    thread.requestInterruption()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )

    def closeEvent(self, event):
        self.request_background_stop()
        if self.has_active_background_tasks():
            QtWidgets.QMessageBox.information(
                self,
                "Background Tasks Running",
                "Solution-space work is still running. Stop or wait for the active computation before closing this view.",
            )
            event.ignore()
            return
        event.accept()

    def on_app_quit(self):
        self.request_background_stop()
