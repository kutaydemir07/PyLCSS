# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Shared helpers for the FEM node package.

Public surface:
    MATERIAL_DATABASE — preset elastic properties.
    suppress_output   — silence Netgen's C-level stdout/stderr.
    OCCGeometry       — Netgen OCC bridge (or None if Netgen is missing).

The in-house scikit-fem FEM solver and the old centroid-based topology
helpers (build_filter_matrix / sensitivity_filter / heaviside_projection /
mma_update / shape_recovery / _assemble_traction_force / …) were removed
along with the solvers that needed them. CalculiX is the trusted FEA
backend and pyMOTO drives the voxel SIMP loop directly.
"""
import os
import sys
import contextlib
import logging

from pylcss.config import simulation_config

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout **and** C-level stdout/stderr.

    Python's sys.stdout redirect does not silence output written directly to
    file-descriptor 1 (e.g. Netgen's C++ std::cout).  This implementation
    uses os.dup2() to redirect the raw file descriptors so that *all* output
    — including C-extension output — is sent to /dev/null (or NUL on Windows).
    """
    if not simulation_config.SUPPRESS_EXTERNAL_LIBRARY_OUTPUT:
        yield
        return

    old_py_stdout = sys.stdout
    old_py_stderr = sys.stderr
    for stream in (old_py_stdout, old_py_stderr):
        flush = getattr(stream, "flush", None)
        if callable(flush):
            try:
                flush()
            except (OSError, ValueError):
                pass

    # ``pythonw.exe`` and frozen windowed applications have no console:
    # sys.stdout/sys.stderr are None and descriptors 1/2 may be unavailable.
    # Redirect every channel that exists without requiring any of them.
    saved_fds = {}
    devnull = open(os.devnull, "w")
    try:
        devnull_fd = devnull.fileno()
        for fd in (1, 2):
            try:
                saved_fd = os.dup(fd)
            except OSError:
                continue
            try:
                os.dup2(devnull_fd, fd)
            except OSError:
                os.close(saved_fd)
            else:
                saved_fds[fd] = saved_fd

        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        for fd, saved_fd in saved_fds.items():
            try:
                os.dup2(saved_fd, fd)
            except OSError:
                pass
            finally:
                os.close(saved_fd)
        sys.stdout = old_py_stdout
        sys.stderr = old_py_stderr
        devnull.close()


try:
    from netgen.occ import OCCGeometry
except ImportError:
    OCCGeometry = None


# Professional Material Database (E in MPa, density in tonne/mm^3)
MATERIAL_DATABASE = {
    'Custom': {'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9, 'k': 45.0},
    'Steel (Structural)': {'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9, 'k': 45.0},
    'Steel (Stainless 304)': {'E': 193000.0, 'nu': 0.29, 'rho': 8.00e-9, 'k': 16.2},
    'Aluminum 6061-T6': {'E': 68900.0, 'nu': 0.33, 'rho': 2.70e-9, 'k': 167.0},
    'Aluminum 7075-T6': {'E': 71700.0, 'nu': 0.33, 'rho': 2.81e-9, 'k': 130.0},
    'Titanium Ti-6Al-4V': {'E': 113800.0, 'nu': 0.34, 'rho': 4.43e-9, 'k': 6.7},
    'Copper (Annealed)': {'E': 110000.0, 'nu': 0.34, 'rho': 8.96e-9, 'k': 401.0},
    'Brass': {'E': 100000.0, 'nu': 0.34, 'rho': 8.50e-9, 'k': 109.0},
    'Cast Iron (Gray)': {'E': 100000.0, 'nu': 0.26, 'rho': 7.20e-9, 'k': 52.0},
    'Magnesium AZ31': {'E': 45000.0, 'nu': 0.35, 'rho': 1.77e-9, 'k': 96.0},
    'Nickel Alloy 718': {'E': 200000.0, 'nu': 0.30, 'rho': 8.19e-9, 'k': 11.4},
    'CFRP (Quasi-Isotropic)': {'E': 70000.0, 'nu': 0.30, 'rho': 1.55e-9, 'k': 6.0},
    'GFRP (E-Glass)': {'E': 25000.0, 'nu': 0.23, 'rho': 1.90e-9, 'k': 0.3},
    'Concrete (Normal)': {'E': 30000.0, 'nu': 0.20, 'rho': 2.40e-9, 'k': 1.7},
    'ABS Plastic': {'E': 2300.0, 'nu': 0.35, 'rho': 1.05e-9, 'k': 0.18},
    'Nylon 6/6': {'E': 2900.0, 'nu': 0.40, 'rho': 1.14e-9, 'k': 0.25},
    'PEEK': {'E': 3600.0, 'nu': 0.38, 'rho': 1.30e-9, 'k': 0.25},
    'Wood (Oak)': {'E': 12000.0, 'nu': 0.35, 'rho': 0.60e-9, 'k': 0.17},
}
