# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.

"""Compatibility façade for the split Design Studio interface."""

from .code_editor import CadCodeEditorDialog
from .execution_workers import GraphExecutionWorker, TopOptStepExportWorker
from .inspector import ExpressionEdit, InspectorSection, PropertiesPanel
from .panels import LibraryPanel, ResultsPanel, TimelinePanel
from .workbench import ProfessionalCadApp, main

__all__ = [
    "CadCodeEditorDialog",
    "ExpressionEdit",
    "GraphExecutionWorker",
    "InspectorSection",
    "LibraryPanel",
    "ProfessionalCadApp",
    "PropertiesPanel",
    "ResultsPanel",
    "TimelinePanel",
    "TopOptStepExportWorker",
    "main",
]

if __name__ == "__main__":
    main()
