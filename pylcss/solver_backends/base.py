# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Core contracts shared by external solver adapters."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections.abc import Callable


_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class SolverBackendError(RuntimeError):
    """A user-actionable failure while preparing, running, or reading a solve."""


def validate_job_name(value: object, *, default: str) -> str:
    """Return a safe solver job name or raise a user-facing error.

    Solver job names become filenames and command-line arguments. Restricting
    them to a portable identifier prevents accidental path traversal and keeps
    CalculiX/OpenRadioss artifact names predictable on every platform.
    """
    name = str(value or default).strip()
    if name in {".", ".."} or not _JOB_NAME_RE.fullmatch(name):
        raise SolverBackendError(
            "Solver job_name must be 1-128 characters and contain only ASCII "
            "letters, digits, dots, underscores, or hyphens."
        )
    return name


@dataclass
class ExternalRunConfig:
    """Runtime options shared by external solver adapters."""

    executable: str | None = None
    secondary_executable: str | None = None
    work_dir: str | None = None
    run_solver: bool = False
    timeout_s: float = 3600.0
    job_name: str = "pylcss_case"
    cancel_callback: Callable[[], bool] | None = None

    def validated_timeout(self) -> float:
        """Return a finite, positive timeout suitable for subprocess APIs."""
        try:
            timeout = float(self.timeout_s)
        except (TypeError, ValueError) as exc:
            raise SolverBackendError("Solver timeout must be a number.") from exc
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise SolverBackendError(
                "Solver timeout must be finite and greater than zero."
            )
        return timeout

    def validated_job_name(self, *, default: str) -> str:
        """Return the configured job name after portable filename validation."""
        return validate_job_name(self.job_name, default=default)
