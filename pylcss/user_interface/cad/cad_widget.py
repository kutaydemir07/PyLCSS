# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.

"""Compatibility façade for the split Design Studio interface."""

from .code_editor import CadCodeEditorDialog
from .execution_workers import (
    GraphExecutionWorker,
    TopOptCadPreviewWorker,
    TopOptStepExportWorker,
)
from .inspector import ExpressionEdit, InspectorSection, PropertiesPanel
from .panels import EventLog, LibraryPanel, ResultsPanel
from .workbench import ProfessionalCadApp, main

__all__ = [
    "CadCodeEditorDialog",
    "EventLog",
    "ExpressionEdit",
    "GraphExecutionWorker",
    "InspectorSection",
    "LibraryPanel",
    "ProfessionalCadApp",
    "PropertiesPanel",
    "ResultsPanel",
    "TopOptCadPreviewWorker",
    "TopOptStepExportWorker",
    "main",
]

if __name__ == "__main__":
    main()
