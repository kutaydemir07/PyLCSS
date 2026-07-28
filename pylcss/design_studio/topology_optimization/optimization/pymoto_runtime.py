# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Thread-safe pyMOTO import without GUI backend side effects."""

from __future__ import annotations

import importlib
import logging
import sys
import threading
from types import ModuleType
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)
_PYMOTO_IMPORT_LOCK = threading.RLock()


class PyMotoDomain(Protocol):
    """Structural interface used from pyMOTO's structured voxel domain."""

    nel: int
    nnodes: int
    dim: int
    elements: np.ndarray
    nodes: np.ndarray

    def get_dofnumber(
        self,
        nodes: np.ndarray,
        dofs: list[int],
        *,
        ndof: int,
    ) -> np.ndarray:
        """Return global degree-of-freedom numbers for the selected nodes."""
        ...


def import_pymoto() -> ModuleType:
    """Import pyMOTO without letting it replace PyLCSS's GUI backend.

    pyMOTO imports all optional plotting modules from its package initializer,
    and the installed release hard-codes ``matplotlib.use("TkAgg")`` there.
    Switching a running Qt application to Tk raises an ImportError and makes
    topology studies depend on which PyLCSS tab was opened first. PyLCSS does
    not use pyMOTO's plotting modules, so retain the active backend.
    """
    with _PYMOTO_IMPORT_LOCK:
        loaded = sys.modules.get("pymoto")
        if loaded is not None:
            return loaded

        import matplotlib

        original_use = matplotlib.use

        def guarded_use(
            backend: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            if str(backend or "").strip().lower() == "tkagg":
                logger.debug(
                    "Ignored pyMOTO TkAgg request; retaining Matplotlib %s.",
                    matplotlib.get_backend(),
                )
                return None
            return original_use(backend, *args, **kwargs)

        matplotlib.use = guarded_use
        try:
            return importlib.import_module("pymoto")
        finally:
            matplotlib.use = original_use
