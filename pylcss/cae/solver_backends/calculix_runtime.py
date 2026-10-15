# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Thread limits for CalculiX child processes."""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)


def calculix_thread_environment(warnings: list[str]) -> dict[str, str]:
    """Select solver threads without changing the application's environment.

    Large inherited OpenMP thread counts can make small and medium FEA models
    substantially slower. Keep a conservative default, with a solver-specific
    override for larger models. BLAS runtimes receive the same limit so an
    inherited library setting cannot independently oversubscribe the solve.
    """
    available = max(1, os.cpu_count() or 1)
    threads = min(4, available)
    setting = os.environ.get("PYLCSS_CALCULIX_THREADS")
    source = "default"
    if setting is not None:
        try:
            requested = int(setting)
            if requested <= 0:
                raise ValueError
        except ValueError:
            message = (
                f"Invalid PYLCSS_CALCULIX_THREADS={setting!r}; expected a positive "
                f"integer. Using {threads} CalculiX thread(s)."
            )
            warnings.append(message)
            logger.warning(message)
        else:
            threads = min(requested, available)
            source = "PYLCSS_CALCULIX_THREADS"
            if requested > available:
                message = (
                    f"PYLCSS_CALCULIX_THREADS={requested} exceeds the {available} "
                    f"available logical CPU(s); using {threads} CalculiX thread(s)."
                )
                warnings.append(message)
                logger.warning(message)

    logger.info("CalculiX: using %d thread(s) (%s).", threads, source)
    return {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    }
