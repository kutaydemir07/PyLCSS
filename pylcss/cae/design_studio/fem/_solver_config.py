# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Execution settings shared by the static and modal FEM nodes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pylcss.input_values import as_bool

if TYPE_CHECKING:
    from pylcss.cae.solver_backends.base import ExternalRunConfig


def add_external_solver_properties(node) -> None:
    """Add the common executable, work-directory, and execution controls."""
    node.create_property("external_solver_path", "", widget_type="text")
    node.create_property("external_work_dir", "", widget_type="text")
    node.create_property("deck_only", False, widget_type="checkbox")
    node.create_property("external_timeout_s", 3600.0, widget_type="float")


def external_run_config(node, *, job_name, cancel_callback=None) -> ExternalRunConfig:
    """Resolve project-relative paths and validate the requested timeout."""
    from pylcss.cae.solver_backends.base import ExternalRunConfig

    project_dir = getattr(node, "_project_dir", None)
    paths = []
    for name in ("external_solver_path", "external_work_dir"):
        value = str(node.get_property(name) or "").strip() or None
        if value and project_dir and not Path(value).is_absolute():
            value = str(Path(project_dir) / value)
        paths.append(value)
    timeout = node.get_property("external_timeout_s")
    config = ExternalRunConfig(
        executable=paths[0],
        work_dir=paths[1],
        run_solver=not as_bool(node.get_property("deck_only")),
        timeout_s=3600.0 if timeout is None else timeout,
        job_name=job_name,
        cancel_callback=cancel_callback,
    )
    config.timeout_s = config.validated_timeout()
    return config
